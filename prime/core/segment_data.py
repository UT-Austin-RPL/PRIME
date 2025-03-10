import numpy as np
import os
import pickle
from tqdm import trange

from prime.configs.primitive_config import get_primitive_config
from prime.utils.data_utils import load_hdf5_demo
from prime.utils.segment_utils import parse_demo_to_primitive_seq, test_with_primitives
import prime.utils.env_utils as EnvUtils

from robosuite import load_controller_config
from robosuite.controllers.skill_controller import SkillController

import robomimic.utils.file_utils as FileUtils


def segment_demos(demo_path, num_demos, device, render, idm_type_model_path, idm_params_model_path, primitive_set,
                  output_mode, controller, segmented_data_dir, save_failed_trajs, parser_algo,
                  max_primitive_horizon, verbose, playback_segmented_trajs):
    segmented_trajs_save_path = os.path.join(segmented_data_dir, 'segmented_trajs.pkl')
    if not os.path.exists(segmented_trajs_save_path):
        demos, env_meta = load_hdf5_demo(demo_path=demo_path, num_trajs=num_demos)
        env_meta['env_kwargs']['controller_configs'] = load_controller_config(default_controller=controller)
        env = EnvUtils.create_env_for_segmentation_evaluation(
            env_meta=env_meta,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_height=84,
            camera_width=84,
            reward_shaping=False,
        )
        primitives_kwargs = dict(
            render=render,
            controller_type=controller,
            image_obs_in_info=False,
            aff_type=None,
            reach_use_gripper=False,
            primitive_set=get_primitive_config(env_meta['env_name']) if primitive_set is None else primitive_set,
            output_mode=output_mode,
        )
        skill_controller = SkillController(env=env, **primitives_kwargs)

        idm_type_model, _ = FileUtils.policy_from_checkpoint(ckpt_path=idm_type_model_path, device=device, verbose=False)
        idm_type_model.start_episode()
        idm_params_model, _ = FileUtils.policy_from_checkpoint(ckpt_path=idm_params_model_path, device=device, verbose=False)
        idm_params_model.start_episode()

        segmented_trajs = []
        num_successful_segmented_trajs = 0
        successful_demos_num = 0
        num_total_segmented_trajs = len(demos)
        segmented_traj_lengths = []
        failed_traj_indices = []

        for ind in trange(len(demos)):
            states = demos[ind]['states']
            initial_state = demos[ind]['initial_state']
            actions = demos[ind]['actions']
            obs = demos[ind]['obs']

            successful_demos_num += int(demos[ind]['playback_succ']['task'])
            if verbose:
                print("Demo index", ind)
                print("Demo len", len(obs))

            segmented_traj, log_prob = parse_demo_to_primitive_seq((obs, list(actions), states), idm_type_model,
                                                                   idm_params_model,
                                                                   max_primitive_horizon=max_primitive_horizon,
                                                                   algo=parser_algo, skill_controller=skill_controller,
                                                                   verbose=verbose)
            segmented_traj_len = len(segmented_traj['seg_ps'])
            segmented_traj_lengths.append(segmented_traj_len)
            if verbose:
                print("Segmented trajectory length", segmented_traj_len)

            if playback_segmented_trajs:
                curr_is_successful = test_with_primitives(env, initial_state, segmented_traj, skill_controller)

                if curr_is_successful:
                    num_successful_segmented_trajs += 1
                    if verbose:
                        print("Successful segmented trajectory!")
                else:
                    failed_traj_indices.append(ind)
                    if verbose:
                        print("Failed segmented trajectory!")
                        print("Failed trajectory indices", failed_traj_indices)

            if (not playback_segmented_trajs) or curr_is_successful or save_failed_trajs:
                segmented_trajs.append(dict(seg_states=segmented_traj['seg_states'],
                                            seg_ps=segmented_traj['seg_ps'],
                                            seg_args=segmented_traj['seg_args'],
                                            original_states=segmented_traj['original_states'],
                                            original_actions=actions,
                                            initial_state=initial_state,
                                            states=states,
                                            is_succ=curr_is_successful))

        print("Number of demonstrations:", num_total_segmented_trajs)
        print("Average length of segmented trajectories", np.mean(segmented_traj_lengths))
        if playback_segmented_trajs:
            print("Playback success rate of demonstrations", successful_demos_num/num_total_segmented_trajs)
            print("Playback success rate of segmented trajectories", num_successful_segmented_trajs/num_total_segmented_trajs)

        save_data = dict(
            env_meta=env_meta,
            traj_info=segmented_trajs
        )
        os.makedirs(os.path.dirname(segmented_trajs_save_path), exist_ok=True)
        with open(segmented_trajs_save_path, "wb") as f_out:
            pickle.dump(save_data, f_out)
    else:
        print(f"Segmented trajectories already exist, loading from {segmented_trajs_save_path}")
        with open(segmented_trajs_save_path, "rb") as f_in:
            save_data = pickle.load(f_in)
    return save_data


