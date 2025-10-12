import argparse
import cv2
import datetime
import h5py
import init_path
import json
import numpy as np
import os
import robosuite as suite
import time
from glob import glob
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.utils.input_utils import input2action


import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *


class CannotFindPathError(Exception):
    pass

class CannotFindValidLocationError(Exception):
    pass


def collect_human_trajectory(
    env, device, arm, env_configuration, problem_info, remove_directory=[]
):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except:
            continue
        
    sim = env.env.env.sim
    real_env = env.env.env
    
    def get_qpos(joint_name):
        qpos_addr = sim.model.get_joint_qpos_addr(joint_name)
        return sim.data.qpos[qpos_addr[0] : qpos_addr[1]].copy()
    
    def find_new_place(obj):
        other_objects = [o for o in real_env.objects_dict.values() if o.name != obj.name]
        x_min = -0.2 - 0.3 + obj.horizontal_radius
        x_max = -0.2 + 0.3 - obj.horizontal_radius
        y_min = 0.0 - 0.6 + obj.horizontal_radius
        y_max = 0.0 + 0.6 - obj.horizontal_radius

        obj_x, obj_y, _ = get_qpos(obj.joints[0])[:3]
        
        for _ in range(500):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            assert len(obj.joints) == 1
            z = get_qpos(obj.joints[0])[2]
            
            location_valid = True
            for other_obj in other_objects:
                assert other_obj.name != obj.name
                assert len(other_obj.joints) == 1
                ox, oy, oz = get_qpos(other_obj.joints[0])[:3]
                if (
                    np.linalg.norm((x - ox, y - oy))
                    <= other_obj.horizontal_radius + obj.horizontal_radius + 0.02
                ) and (
                    z - z <= other_obj.top_offset[-1] - obj.bottom_offset[-1]
                ):
                    location_valid = False
                    break
                
            if np.linalg.norm((x - obj_x, y - obj_y)) <= 0.3:
                location_valid = False
                break

            if location_valid:
                break
                
        if not location_valid:
            raise CannotFindValidLocationError("Cannot find a valid location to place the object")
            
        return (x, y, z)

    def find_path(interest_obj, new_x, new_y, show_animation=False):
        scale = 100.0
        import sys
        sys.path.append("./PythonRobotics/PathPlanning/RRTStar/")
        from rrt_star import RRTStar
        other_objects = [o for o in real_env.objects_dict.values() if o.name != interest_obj.name]
        assert len(other_objects) == len(real_env.objects_dict) - 1
        obstacle_list = []
        for other_obj in other_objects:
            assert len(other_obj.joints) == 1
            ox, oy, oz = get_qpos(other_obj.joints[0])[:3]
            obstacle_list.append((ox * scale, oy * scale, scale * (other_obj.horizontal_radius + 0.02)))
        
        rrt_star = RRTStar(
            start=get_qpos(interest_obj.joints[0])[:2] * scale,
            goal=np.array([new_x, new_y]) * scale,
            rand_area=np.array([-0.2 - 0.3, -0.2 + 0.3, 0.0 - 0.6, 0.0 + 0.6]) * scale,
            obstacle_list=obstacle_list,
            expand_dis=30,
            robot_radius=(interest_obj.horizontal_radius + 0.02) * scale,
            # path_resolution=0.05,
            max_iter=1000,
        )

        path = rrt_star.planning(animation=show_animation)
        
        if path is None:
            raise CannotFindPathError("Cannot find a path for the object to move")
        else:
            print("found path with length", len(path))
        
        if show_animation:
            import matplotlib.pyplot as plt
            rrt_star.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.show()
        
        return np.array(path) / scale
    
    sim = env.env.env.sim
    interest_obj_name = 'akita_black_bowl_1'
    interest_obj = env.env.env.objects_dict[interest_obj_name]
    
    new_x, new_y, _ = find_new_place(interest_obj)
    path = find_path(interest_obj, new_x, new_y)
    
    distance = np.linalg.norm(path[1:] - path[:-1], axis=1).sum()
    steps = 60
    velocity = distance / steps

    frame_path = []
    for i in range(len(path)-1):
        n_points = int(np.linalg.norm(path[i+1] - path[i]) / velocity)
        xs = np.linspace(path[i][0], path[i+1][0], n_points)
        ys = np.linspace(path[i][1], path[i+1][1], n_points)
        if i > 0:
            xs = xs[1:]
            ys = ys[1:]
        frame_path.append(np.stack([xs, ys], axis=1))
    frame_path = np.concatenate(frame_path, axis=0)
    
    # ID = 2 always corresponds to agentview
    env.render()

    task_completion_hold_count = (
        -1
    )  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    # Loop until we get a reset from the input or the task completes
    saving = True
    count = 0
    
    while True:
        if 5 <= count < len(frame_path) + 5:
            for joint in interest_obj.joints:
                qpos_addr = sim.model.get_joint_qpos_addr(joint)
                qpos = sim.data.qpos[qpos_addr[0] : qpos_addr[1]].copy()
                qpos[:2] = frame_path[count - 5]
                sim.data.qpos[qpos_addr[0] : qpos_addr[1]] = qpos

        count += 1
        # Set active robot
        active_robot = (
            env.robots[0]
            if env_configuration == "bimanual"
            else env.robots[arm == "left"]
        )

        # Get the newest action
        action, grasp = input2action(
            device=device,
            robot=active_robot,
            active_arm=arm,
            env_configuration=env_configuration,
        )

        # If action is none, then this a reset so we should break
        if action is None:
            print("Break")
            saving = False
            break

        # Run environment step

        env.step(action)
        env.render()
        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    print(count)
    # cleanup for end of data collection episodes
    if not saving:
        remove_directory.append(env.ep_directory.split("/")[-1])
    env.close()
    return saving


def gather_demonstrations_as_hdf5(
    directory, out_dir, env_info, args, remove_directory=[]
):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        # print(ep_directory)
        if ep_directory in remove_directory:
            # print("Skipping")
            continue
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])

        if len(states) == 0:
            continue

        # Delete the first actions and the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    grp.attrs["problem_info"] = json.dumps(problem_info)
    grp.attrs["bddl_file_name"] = args.bddl_file
    grp.attrs["bddl_file_content"] = str(open(args.bddl_file, "r", encoding="utf-8"))

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default="demonstration_data",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Which camera to use for collecting demos",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="OSC_POSE",
        help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'",
    )
    parser.add_argument("--device", type=str, default="spacemouse")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.5,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--num-demonstration",
        type=int,
        default=50,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument("--bddl-file", type=str)

    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50734)

    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # Create argument configuration
    config = {
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    assert os.path.exists(args.bddl_file)
    problem_info = BDDLUtils.get_problem_info(args.bddl_file)
    # Check if we're using a multi-armed environment and use env_configuration argument if so

    # Create environment
    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]
    if "TwoArm" in problem_name:
        config["env_configuration"] = args.config
    print(language_instruction)
    env = TASK_MAPPING[problem_name](
        bddl_file_name=args.bddl_file,
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "demonstration_data/tmp/{}_ln_{}/{}".format(
        problem_name,
        language_instruction.replace(" ", "_").strip('""'),
        str(time.time()).replace(".", "_"),
    )

    env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity
        )
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            args.vendor_id,
            args.product_id,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(
        args.directory,
        f"{domain_name}_ln_{problem_name}_{t1}_{t2}_"
        + language_instruction.replace(" ", "_").strip('""'),
    )

    os.makedirs(new_dir)

    # collect demonstrations

    remove_directory = []
    i = 0
    while i < args.num_demonstration:
        print(i)
        try:
            saving = collect_human_trajectory(
                env, device, args.arm, args.config, problem_info, remove_directory
            )
        except (CannotFindPathError, CannotFindValidLocationError) as e:
            print(e)
            continue

        if saving:
            print(remove_directory)
            gather_demonstrations_as_hdf5(
                tmp_directory, new_dir, env_info, args, remove_directory
            )
            i += 1
