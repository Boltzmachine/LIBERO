import argparse
import cv2
import datetime
import h5py
import init_path
import pandas as pd
import json
import numpy as np
import os
import robosuite as suite
import time
from glob import glob
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.utils.input_utils import input2action
from copy import deepcopy

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
    
class RobotAutoController:
    def __init__(self, env, robot):
        self.stage = "move_to_obj" # "move_to_obj", "move_down1", "grasp", "move_up", "move_to_target", "move_down2", "release"
        self.env = env
        self.robot = robot
        self.robot_init_pos = robot.controller.ee_pos.copy()
        self.task = None
        self.top_z = None
        
        self.moving_obj_names = []
        for moving_obj_name in env.obj_of_interest[::-1]:
            if moving_obj_name not in self.moving_obj_names:
                self.moving_obj_names.append(moving_obj_name)
        self.advance_to_next()
        
        self._finish = False
        
    def advance_to_next(self):
        """advance to the next moving object"""
        if len(self.moving_obj_names) == 0:
            self._finish = True
            return False
        while True:
            next_obj_name = self.moving_obj_names.pop(0)
            if next_obj_name in self.env.objects_dict:
                self.curr_moving_obj = self.env.objects_dict[next_obj_name]
                break
        self.stage = "move_to_obj"
        return True
    
    def set_task(self, task, **task_kwargs):
        self.task = task
        if task is None:
            return
        if task == "grab_and_move":
            self.set_goal(task_kwargs["goal_pos"])
            self.interest_pos = self.env.get_qpos(task_kwargs['target_object'])[:3]
            self.restore_after_complete = task_kwargs.get("restore_after_complete", True)
        elif task == "wait":
            self.wait_steps = task_kwargs['wait_steps']
        elif task == "dead":
            pass
        else:
            raise NotImplementedError(f"Unknown task {task}")
        
    def step(self):
        if self.task is None or self.task == "dead":
            return None
        func = getattr(self, self.task)
        return func()
    
    def set_goal(self, goal_pos):
        if goal_pos is None:
            self.goal_pos = env.object_states_dict[self.curr_moving_obj.name].goal_pos
        else:
            self.goal_pos = goal_pos
            
    def wait(self):
        self.wait_steps -= 1
        if self.wait_steps <= 0:
            return "finish"
        return None
        
    def grab_and_move(self):
        if self._finish:
            return None
        next_key = None
        step_size = 0.012 #device._pos_step * device.pos_sensitivity
        
        robot = self.robot
        eef_pos = robot.controller.ee_pos.copy()
        
        interest_pos = self.interest_pos
        goal_pos = self.goal_pos
        
        if self.stage == "move_to_obj" or self.stage == "move_down1":
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
            if next_key is None and self.stage != "move_down1":
                self.stage = "move_down1"
                
        if self.stage == "move_down1":
            if next_key is None:
                self.top_z = self.env.get_qpos(self.curr_moving_obj)[2] + self.curr_moving_obj.top_offset[-1]
                if eef_pos[2] > self.top_z + 0.001:
                    next_key = 'f'
                
            if next_key is None:
                self.stage = "grasp"
                next_key = Key.space
                
        if self.stage == "grasp":
            if np.isclose(robot.gripper.current_action, [-1, 1]).all():
                self.stage = "move_up"
                
        if self.stage == "move_up":
            if eef_pos[2] < self.robot_init_pos[2]:
                next_key = 'r'
            if next_key is None:
                self.stage = "move_to_target"
        
        if self.stage == "move_to_target" or self.stage == "move_down2":
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
                    
            if next_key is None and self.stage != "move_down2":
                self.stage = "move_down2"
                
        if self.stage == "move_down2":
            if eef_pos[2] > self.top_z + 0.05:
                next_key = 'f'
            if next_key is None:
                self.top_z = None
                self.stage = "release"
                next_key = Key.space
                
        if self.stage == "release":
            if np.isclose(robot.gripper.current_action, [1, -1]).all():
                self.stage = "move_up2"
        
        if self.stage == "move_up2":
            if self.restore_after_complete and eef_pos[2] < self.robot_init_pos[2]:
                next_key = 'r'
                
            if next_key is None:
                self.stage = "move_to_obj"
                return "finish"
                self.advance_to_next()

        return next_key


def collect_human_trajectory(
    env, device, arm, env_configuration, problem_info, remove_directory=[], save_failed=False, predictive_control_data=None
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

    active_robot = (
        env.robots[0]
        if env_configuration == "bimanual"
        else env.robots[arm == "left"]
    )
    robot_controller = RobotAutoController(real_env, active_robot)
    
    auto_control = True
    max_count = 720 if auto_control else float("inf")
    success = False
    
    first_moved = False
    second_moved = False
    while True:
        eef_pos = active_robot.controller.ee_pos.copy()
        
        next_key = None
        if robot_controller.task is None:
            first_object_original_pos = real_env.get_qpos(real_env.objects_dict['tomato_sauce_1'])[:3]
            robot_controller.set_task("grab_and_move", goal_pos=real_env.sim.data.body_xpos[real_env.sim.model.body_name2id('flat_stove_1_burner_plate')], target_object=real_env.objects_dict['tomato_sauce_1'], restore_after_complete=False)
        next_key = robot_controller.step()
        if next_key == "finish":
            if robot_controller.task == "grab_and_move":
                if second_moved:
                    robot_controller.set_task("dead")
                elif first_moved:
                    robot_controller.set_task("grab_and_move", goal_pos=real_env.sim.data.body_xpos[real_env.sim.model.body_name2id('flat_stove_1_burner_plate')], target_object=real_env.objects_dict['alphabet_soup_1'])
                    second_moved = True
                else:
                    robot_controller.set_task("wait", wait_steps=20)
                    first_moved = True
            elif robot_controller.task == "wait":
                robot_controller.set_task("grab_and_move", goal_pos=first_object_original_pos, target_object=real_env.objects_dict['tomato_sauce_1'])
            else:
                print("Unknown task finish")
        
        if next_key is not None and auto_control:
            if isinstance(next_key, str):
                next_key = KeyCode.from_char(next_key)
            else:
                assert isinstance(next_key, Key)
            device.on_press(next_key)
            device.on_release(next_key)

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
        
        diff_eef_pos = active_robot.controller.ee_pos - eef_pos
        if predictive_control_data is not None and next_key is not None:
            if hasattr(next_key, 'char'):
                key_str = next_key.char
                if key_str in ['w', 'a', 's', 'd', 'r', 'f']:
                    predictive_control_data.append({
                        "eef_pos": deepcopy(eef_pos),
                        "key": key_str,
                        "diff_eef_pos": diff_eef_pos,
                    })
        
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

        if count >= max_count and task_completion_hold_count < 0:
            # timeout
            saving = False
            break
        
    info = {
        "success": success,
        "length": count,
        "extra_states": real_env._get_extra_states(),
    }
    # cleanup for end of data collection episodes
    if not saving and not save_failed:
        remove_directory.append(env.ep_directory.split("/")[-1])
    else:
        np.save(
            os.path.join(env.ep_directory, "extra_info.npy"),
            info,
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
        
        extra_info = np.load(os.path.join(directory, ep_directory, "extra_info.npy"), allow_pickle=True).item()
        
        def create_dataset_from_dict(grp, key, data):
            assert isinstance(data, dict)
            for key in data:
                if isinstance(data[key], dict):
                    sub_grp = grp.create_group(key)
                    create_dataset_from_dict(sub_grp, key, data[key])
                if isinstance(data[key], np.ndarray):
                    grp.create_dataset(key, data=data[key])

        create_dataset_from_dict(ep_data_grp, None, extra_info) #FIXME: no success?

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
    
    predictive_control_data = []
    save_freq = 10

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
                    env, device, args.arm, args.config, problem_info, remove_directory, save_failed=save_failed, predictive_control_data=predictive_control_data
                )
            except (CannotFindPathError, CannotFindValidLocationError) as e:
                print(e)
                env.close()
                continue
            break
        successes.append(info["success"])
        print("Success rate: {}/{} ({:.2f}%)".format(sum(successes), len(successes), sum(successes)/len(successes)*100))
        if info["success"]:
            lengths.append(info["length"])

        i += 1
        if (i + 1) % save_freq == 0:
            if saving or save_failed:
                gather_demonstrations_as_hdf5(
                    tmp_directory, new_dir, env_info, args, remove_directory
                )
                
                if predictive_control_data is not None:
                    np.save(
                        "predictive_control_data.npy",
                        predictive_control_data,
                        allow_pickle=True
                    )
            
    print("Length: mean {}, max {}, len {}".format(np.mean(lengths), np.max(lengths), len(lengths)))
