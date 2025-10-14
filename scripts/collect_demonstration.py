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
from libero.libero.utils.errors import CannotFindPathError, CannotFindValidLocationError

from robosuite.devices import Keyboard
from pynput.keyboard import KeyCode, Key


class ArrowKeyboard(Keyboard):
    def _translate(self, key):
        if hasattr(key, 'char'):
            if key.char == '4':
                key = KeyCode.from_char('a')
            elif key.char == '6':
                key = KeyCode.from_char('d')
            elif key.char == '8':
                key = KeyCode.from_char('w')
            elif key.char == '2':
                key = KeyCode.from_char('s')
        else:
            if key == Key.up:  # Up arrow
                key = KeyCode.from_char('r')
            elif key == Key.down:  # Down arrow
                key = KeyCode.from_char('f')
        return key
    
    def on_press(self, key):
        key = self._translate(key)
        return super().on_press(key)

    def on_release(self, key):
        key = self._translate(key)
        return super().on_release(key)

def collect_human_trajectory(
    env, device, arm, env_configuration, problem_info, remove_directory=[], save_failed=False
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
            env.init_moving_params()
            reset_success = True
        except Exception as e:
            print(e)
            continue
    
    real_env = env.env.env
    
    # ID = 2 always corresponds to agentview
    env.render()

    task_completion_hold_count = (
        -1
    )  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    # Loop until we get a reset from the input or the task completes
    saving = True
    count = 0

    robot_stage = "move_to_obj" # "move_to_obj", "move_down1", "grasp", "move_up", "move_to_target", "move_down2", "release"

    active_robot = (
        env.robots[0]
        if env_configuration == "bimanual"
        else env.robots[arm == "left"]
    )
    robot_init_pos = active_robot.controller.ee_pos.copy()
    goal_pos = real_env.object_states_dict[real_env.obj_of_interest[0]].goal_pos
    
    success = False
    while True:
        next_key = None

        if count > len(real_env.frame_path) + 5:
            step_size = 0.01 #device._pos_step * device.pos_sensitivity
            eef_pos = active_robot.controller.ee_pos
            interest_pos = real_env.get_qpos(real_env.interest_obj)[:3]
            if robot_stage == "move_to_obj" or robot_stage == "move_down1":
                if eef_pos[0] < interest_pos[0]:
                    if eef_pos[0] + step_size < interest_pos[0] - step_size / 2:
                        next_key = 's'
                if eef_pos[0] > interest_pos[0]:
                    if eef_pos[0] - step_size > interest_pos[0] + step_size / 2:
                        next_key = 'w'
                if eef_pos[1] < interest_pos[1]:
                    if eef_pos[1] + step_size < interest_pos[1] - step_size / 2:
                        next_key = 'd'
                if eef_pos[1] > interest_pos[1]:
                    if eef_pos[1] - step_size > interest_pos[1] + step_size / 2:
                        next_key = 'a'
                if next_key is None and robot_stage != "move_down1":
                    robot_stage = "move_down1"
                    
            if robot_stage == "move_down1":
                if next_key is None:
                    top_z = real_env.get_qpos(real_env.interest_obj)[2] + real_env.interest_obj.top_offset[-1]
                    if eef_pos[2] > top_z + 0.002:
                        next_key = 'f'
                    
                if next_key is None:
                    robot_stage = "grasp"
                    next_key = Key.space
                    
            if robot_stage == "grasp":
                if np.isclose(active_robot.gripper.current_action, [-1, 1]).all():
                    robot_stage = "move_up"
                    
            if robot_stage == "move_up":
                if eef_pos[2] < robot_init_pos[2]:
                    next_key = 'r'
                if next_key is None:
                    robot_stage = "move_to_target"
            
            if robot_stage == "move_to_target" or robot_stage == "move_down2":
                if eef_pos[0] < goal_pos[0]:
                    if eef_pos[0] + step_size < goal_pos[0] - step_size / 2:
                        next_key = 's'
                if eef_pos[0] > goal_pos[0]:
                    if eef_pos[0] - step_size > goal_pos[0] + step_size / 2:
                        next_key = 'w'
                if eef_pos[1] < goal_pos[1]:
                    if eef_pos[1] + step_size < goal_pos[1] - step_size / 2:
                        next_key = 'd'
                if eef_pos[1] > goal_pos[1]:
                    if eef_pos[1] - step_size > goal_pos[1] + step_size / 2:
                        next_key = 'a'
                        
                if next_key is None and robot_stage != "move_down2":
                    robot_stage = "move_down2"
                    
            if robot_stage == "move_down2":
                if eef_pos[2] > top_z + 0.05:
                    next_key = 'f'
                if next_key is None:
                    robot_stage = "release"
                    next_key = Key.space
            

            if next_key is not None:
                if isinstance(next_key, str):
                    next_key = KeyCode.from_char(next_key)
                else:
                    assert isinstance(next_key, Key)
                device.on_press(next_key)
                device.on_release(next_key)
            # else:
            #     import ipdb; ipdb.set_trace()

        count += 1
        
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
            success = True
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

        if count == 380 and task_completion_hold_count < 0:
            # timeout
            saving = False
            break
        
    info = {
        "success": success,
        "length": count,
    }
    # cleanup for end of data collection episodes
    if not saving and not save_failed:
        remove_directory.append(env.ep_directory.split("/")[-1])
    else:
        np.savez(
            os.path.join(env.ep_directory, "extra_info.npz"),
            goal_pos=goal_pos,
            # goal_quat=None,
            **info,
        )
        
    env.close()

    return saving, info


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
        
        extra_info = np.load(os.path.join(directory, ep_directory, "extra_info.npz"), allow_pickle=True)
        for key in extra_info:
            if extra_info[key] is not None:
                ep_data_grp.create_dataset(key, data=extra_info[key])

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
        device = ArrowKeyboard(
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
    save_failed = True
    remove_directory = []
    i = 0
    successes = []
    lengths = []
    while i < args.num_demonstration:
        print("Collecting demonstration {}/{}".format(i + 1, args.num_demonstration))
        while True:    
            try:
                saving, info = collect_human_trajectory(
                    env, device, args.arm, args.config, problem_info, remove_directory, save_failed=save_failed
                )
            except (CannotFindPathError, CannotFindValidLocationError) as e:
                print(e)
                env.close()
                continue
            break
        successes.append(info["success"])
        if info["success"]:
            lengths.append(info["length"])

        if saving or save_failed:
            if save_failed:
                save_dir = os.path.join(new_dir, "success" if info["success"] else "failed")
                os.makedirs(save_dir, exist_ok=True)
            else:
                save_dir = new_dir
            gather_demonstrations_as_hdf5(
                tmp_directory, save_dir, env_info, args, remove_directory
            )
            i += 1
    print("Success rate: {}".format(np.mean(successes)))
    print("Length: mean {}, max {}, len {}".format(np.mean(lengths), np.max(lengths), len(lengths)))
