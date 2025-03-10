from copy import deepcopy
import h5py
import json
import numpy as np
import os
import pickle
import random
from tqdm import trange

from robosuite.controllers.skill_controller import SkillController

from prime.configs.primitive_config import get_primitive_config
from prime.utils.segment_utils import get_pred_params
import prime.utils.env_utils as EnvUtils

import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase
from robomimic.scripts.split_train_val import split_train_val_from_hdf5

def extract_trajectory_with_obs(
    env,
    initial_state,
    states,
    actions,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
    """
    assert isinstance(env, EnvBase)
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    obs = env.reset_to(initial_state)

    traj = dict(
        obs=[],
        next_obs=[],
    )
    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in range(1, traj_len + 1):

        # get next observation
        if t == traj_len:
            # play final action to get next observation for last timestep
            next_obs, _, _, _ = env.step(actions[t - 1])
            last_state = env.get_state()
        else:
            # reset to simulator state to get observation
            next_obs = env.reset_to({"states" : states[t]})

        # collect transition
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)

        # update for next iter
        obs = deepcopy(next_obs)

    ret_obs = traj["obs"] + [traj["next_obs"][-1]]

    return ret_obs, env.is_success(), last_state


def load_hdf5_demo(demo_path, num_trajs=None):
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=demo_path)
    env = EnvUtils.create_env_for_segmentation_evaluation(
        env_meta=env_meta,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_height=84,
        camera_width=84,
        reward_shaping=False,
    )

    ret_list = []
    f = h5py.File(demo_path, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]
    if num_trajs is not None:
        demos = demos[:num_trajs]
    for ind in trange(len(demos)):
        ep = demos[ind]
        states = f["data/{}/states".format(ep)][()]
        model_file = f["data/{}".format(ep)].attrs["model_file"]
        initial_state = dict(states=states[0], model=model_file)
        actions = f["data/{}/actions".format(ep)][()]
        obs, playback_succ, last_state = extract_trajectory_with_obs(
            env=env,
            initial_state=initial_state,
            states=states,
            actions=actions,
        )
        states = np.concatenate([states, np.expand_dims(last_state['states'], axis=0)], axis=0)
        ret_list.append(dict(
            states=states,
            initial_state=initial_state,
            actions=actions,
            obs=obs,
            playback_succ=playback_succ,
        ))
    f.close()
    return ret_list, env_meta


def check_save_flag_for_push_primitive(start_obs, end_obs, seg):
    start_obj_position = start_obs['obj_pos'].reshape(-1, 3)
    end_obj_position = end_obs['obj_pos'].reshape(-1, 3)
    num_objects = start_obj_position.shape[0]
    end_eff_position = end_obs['robot0_eef_pos']
    eef_displacement = seg['args'][-3:-1]
    save_flag = False
    for obj_idx in range(num_objects):
        obj_distance = np.linalg.norm(end_obj_position[obj_idx] - start_obj_position[obj_idx])
        end_eef_to_obj_distance = np.linalg.norm(end_obj_position[obj_idx] - end_eff_position)
        obj_xy_displacement = end_obj_position[obj_idx][:2] - start_obj_position[obj_idx][:2]
        end_eef_to_obj_xy_displacement = end_obj_position[obj_idx][:2] - end_eff_position[:2]
        obj_cos_similarity = np.inner(eef_displacement, obj_xy_displacement) / (
                    np.linalg.norm(eef_displacement) * np.linalg.norm(obj_xy_displacement))
        eef_obj_cos_similarity = np.inner(eef_displacement, end_eef_to_obj_xy_displacement) / (
                    np.linalg.norm(eef_displacement) * np.linalg.norm(end_eef_to_obj_xy_displacement))
        if (
            obj_distance > 0.07 and  # Significant object displacement
            end_eef_to_obj_distance < 0.1 and  # End effector is close to object at the end state
            obj_cos_similarity > np.sqrt(3) / 2 and  # Alignment with object direction
            eef_obj_cos_similarity > 0.1  # Alignment with end effector direction
        ):
            save_flag = True
    return save_flag


def reformat_rollout_data(data_dir, output_mode, controller, env_name, primitive_set, model_type, num_seeds,
                                       num_others_per_traj, val_ratio, camera_names, camera_height, camera_width):

    assert model_type in ['idm_p_type', 'idm_p_params', 'policy_pt_p_type', 'policy_pt_p_params']
    skill_controller = SkillController(env=None, output_mode=output_mode, controller_type=controller,
                                       primitive_set=get_primitive_config(env_name) if primitive_set is None else
                                       primitive_set)
    data_path = os.path.join(data_dir, env_name)
    output_path = os.path.join(data_path, f"{model_type}_training_dataset.hdf5")
    if os.path.exists(output_path):
        print(f"{output_path} already exists! Skip generating training data for {model_type}")
        return
    else:
        print(f"Generating training data for {model_type}")
    assert not os.path.exists(output_path), "{} already exists".format(output_path)
    env_meta_file = '{}/env_meta.pkl'.format(data_path)
    with open(env_meta_file, 'rb') as env_f:
        load_pkl = pickle.load(env_f)
        env_meta = load_pkl["env_meta"]
    data_env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=camera_names,
        camera_height=camera_height,
        camera_width=camera_width,
        reward_shaping=False,
    )

    skill_names = skill_controller.primitive_set
    demos = []
    num_trajs = {name: 0 for name in skill_names}
    if 'p_params' in model_type:
        num_trajs.update({name + '_aug': 0 for name in skill_names})
    elif 'p_type' in model_type:
        num_trajs.update({"atomic": 0, "hard_negative": 0})

    for seed in range(num_seeds):
        if 'p_type' in model_type:  # hard negative samples
            prev_tag = None
            hard_negative_p_list = deepcopy(skill_names)
            hard_negative_p_list.remove('atomic')
            hard_negative_segments = {}
            for _p_name in hard_negative_p_list:
                hard_negative_segments[_p_name] = []
        dataset_filename = '{}/data_{}.pkl'.format(data_path, seed)
        print("Loading data from", dataset_filename)
        with open(dataset_filename, 'rb') as f:
            try:
                while True:
                    load_pkl = pickle.load(f)
                    saved_segments = {}
                    if load_pkl['tag'] in skill_names and load_pkl['interest']:
                        saved_segments["tag"], saved_segments["args"] = load_pkl["tag"], load_pkl["args"]
                        start_ind, end_ind = 0, -1
                        if load_pkl['done']:
                            try:
                                start_obs = data_env.reset_to({"states": load_pkl["state"][start_ind]["states"]})
                                end_obs = data_env.reset_to({"states": load_pkl["state"][end_ind]["states"]})
                            except:
                                start_obs = data_env.reset_to({"states": load_pkl["state"][start_ind]})
                                end_obs = data_env.reset_to({"states": load_pkl["state"][end_ind]})

                            saved_segments['obs'] = [start_obs, end_obs]
                            if load_pkl['tag'] == 'push':  # filter out push primitive rollouts with no object displacement
                                if not check_save_flag_for_push_primitive(start_obs, end_obs, load_pkl):
                                    continue

                            num_trajs[load_pkl['tag']] += 1
                            demos.append(saved_segments) # keys: obs (list), action (list), initial_state, task_xml, tag, args, interest
                            if 'p_type' in model_type:
                                if saved_segments['tag'] != prev_tag and saved_segments['tag'] in hard_negative_p_list:
                                    hard_negative_segments[load_pkl['tag']].append(saved_segments['obs'])
                                prev_tag = saved_segments['tag']
                    elif load_pkl['tag'] == "combine" and 'p_type' in model_type:
                        saved_segments['tag'] = 'atomic'
                        num_trajs['atomic'] += num_others_per_traj
                        # data_env.reset_to(load_pkl['initial_state'])
                        invalid_end_idx = [[] for _ in range(len(load_pkl['state']))]
                        for seg in load_pkl['primitive']:
                            for ob_idx in range(seg['action_start_idx'], seg['action_start_idx'] + seg['p_len']):
                                invalid_end_idx[ob_idx].append(seg['action_start_idx'] + seg['p_len'])

                        for _ in range(num_others_per_traj):
                            for _ in range(20):
                                start_idx = random.randint(0, len(load_pkl['state']) - 6)
                                length = random.randint(5, len(load_pkl['state']) - start_idx)
                                end_idx = min(start_idx + length, len(load_pkl['state'])) - 1
                                if not (end_idx in invalid_end_idx[start_idx]):
                                    try:
                                        start_obs = data_env.reset_to({"states": load_pkl["state"][start_idx]["states"]})
                                    except:
                                        start_obs = data_env.reset_to({"states": load_pkl["state"][start_idx]})
                                    try:
                                        end_obs = data_env.reset_to({"states": load_pkl["state"][end_idx]["states"]})
                                    except:
                                        end_obs = data_env.reset_to({"states": load_pkl["state"][end_idx]})
                                    saved_segments['obs'] = [start_obs, end_obs]
                                    demos.append(saved_segments)
                                    break
                        if 'p_type' in model_type:
                            prev_tag = None
                    elif load_pkl['tag'] == "combine" and 'p_params' in model_type:
                        max_aug_step = 10
                        for seg in load_pkl['primitive']:
                            if seg['p_name'] != 'atomic' and seg['interest'] and seg['done'] and seg['p_name'] != 'push':
                                saved_segments = {'tag': seg['p_name'], 'args': seg['args']}
                                start_idx_center = seg['action_start_idx']
                                end_idx_center = seg['action_start_idx'] + seg['p_len']
                                start_idx = random.randint(max(0, start_idx_center - max_aug_step),
                                                           min(len(load_pkl["state"]) - 1,
                                                               start_idx_center + max_aug_step))
                                end_idx = random.randint(max(0, end_idx_center - max_aug_step),
                                                         min(len(load_pkl["state"]) - 1,
                                                             end_idx_center + max_aug_step))
                                try:
                                    start_obs = data_env.reset_to({"states": load_pkl["state"][start_idx]["states"]})
                                except:
                                    start_obs = data_env.reset_to({"states": load_pkl["state"][start_idx]})
                                try:
                                    end_obs = data_env.reset_to({"states": load_pkl["state"][end_idx]["states"]})
                                except:
                                    end_obs = data_env.reset_to({"states": load_pkl["state"][end_idx]})
                                if seg['p_name'] == 'push':
                                    if not check_save_flag_for_push_primitive(start_obs, end_obs, seg):
                                        continue
                                saved_segments['obs'] = [start_obs, end_obs]
                                demos.append(saved_segments)
                                num_trajs[saved_segments['tag'] + '_aug'] += 1
            except:
                pass
    if 'p_type' in model_type:
        for _p_name in hard_negative_p_list:
            if len(hard_negative_segments[_p_name]) == 0:
                continue
            for _ in range(num_trajs['atomic'] // len(hard_negative_p_list) // 3):
                saved_segments['tag'] = 'atomic'
                idx_1 = random.randint(0, len(hard_negative_segments[_p_name]) - 1)
                idx_2 = random.randint(0, len(hard_negative_segments[_p_name]) - 1)
                if idx_1 == idx_2:
                    continue
                num_trajs['hard_negative'] += 1
                saved_segments['obs'] = [hard_negative_segments[_p_name][idx_1][0], hard_negative_segments[_p_name][idx_2][1]]
                demos.append(saved_segments)
    print(num_trajs)
    print("Finish loading data from ", data_path)

    f_out = h5py.File(output_path, "w") # edit mode
    try:
        data_grp = f_out.create_group("data")
        print("input dir: {}".format(data_path))
        print("output file: {}".format(output_path))
        total_samples = 0
        for ind in trange(len(demos)):
            ep = demos[ind]
            ep_data_grp = data_grp.create_group("demo_{}".format(ind))
            ep_data_grp.attrs["num_samples"] = 1
            ep_data_grp.create_dataset("states", data=np.array([]))
            ep_data_grp.create_dataset("rewards", data=np.array([1.]))
            ep_data_grp.create_dataset("dones", data=np.array([int(True)]))
            start_obs, p, end_obs = ep['obs'][0], ep['tag'], ep['obs'][-1]
            if 'p_params' in model_type:
                norm_args = ep['args']
                norm_output = skill_controller.args_to_output(p_name=p, skill_args=norm_args)
                args_masks = skill_controller.get_skill_param_dim(p_name=p)['mask']
            p_onehot = np.zeros(len(skill_names))
            p_id = skill_names.index(p)
            p_onehot[p_id] = 1.
            if 'p_params' in model_type:
                ep_data_grp.create_dataset("actions", data=np.expand_dims(np.array(norm_output), axis=0))
                ep_data_grp.create_dataset("action_masks", data=np.expand_dims(np.array(args_masks), axis=0))
            elif 'p_type' in model_type:
                ep_data_grp.create_dataset("actions", data=np.expand_dims(p_onehot, axis=0))
            else:
                raise NotImplementedError
            ep_data_grp.create_dataset("obs/primitive_type", data=np.expand_dims(p_onehot, axis=0))
            for k in start_obs:
                if 'idm' in model_type:
                    ep_data_grp.create_dataset("obs/cur_{}".format(k), data=np.expand_dims(np.array(start_obs[k]), axis=0),
                                               compression='gzip')
                    ep_data_grp.create_dataset("obs/goal_{}".format(k), data=np.expand_dims(np.array(end_obs[k]), axis=0),
                                               compression='gzip')
                elif 'policy' in model_type:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.expand_dims(np.array(start_obs[k]), axis=0),
                                               compression='gzip')


            total_samples += 1
        data_grp.attrs["total"] = total_samples
    finally:
        f_out.close()
    split_train_val_from_hdf5(hdf5_path=output_path, val_ratio=val_ratio)


def reformat_segmented_trajs(segmented_data, segmented_data_dir, num_augmentation_per_segment, idm_params_model_path, model_type,
                             controller, output_mode, primitive_set, device, val_ratio, camera_names, camera_height, camera_width):
    assert model_type in ['policy_p_type', 'policy_p_params']
    if idm_params_model_path is not None:
        idm_params_model, _ = FileUtils.policy_from_checkpoint(ckpt_path=idm_params_model_path, device=device, verbose=False)
        idm_params_model.start_episode()

    env_meta = segmented_data['env_meta']
    segmented_trajs = segmented_data['traj_info']

    data_env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=camera_names,
        camera_height=camera_height,
        camera_width=camera_width,
        reward_shaping=False,
    )
    eval_env = EnvUtils.create_env_for_policy_evaluation(
        env_meta=env_meta,
        camera_names=camera_names,
        camera_height=camera_height,
        camera_width=camera_width,
        reward_shaping=False,
    )
    skill_controller = SkillController(env=data_env, output_mode=output_mode, controller_type=controller,
                                       primitive_set=get_primitive_config(env_meta['env_name']) if primitive_set is None else primitive_set,)
    skill_names = skill_controller.primitive_set
    output_dim = skill_controller.output_dim

    output_path = os.path.join(segmented_data_dir, f"{model_type}_training_dataset.hdf5")
    if os.path.exists(output_path):
        print(f"File {output_path} already exists, skipping")
        return
    print(f"Writing to {output_path}")
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    data_grp.attrs["env_args"] = json.dumps(env_meta, indent=4)
    total_samples = 0
    demo_cnt = 0
    for idx in trange(len(segmented_trajs)):
        ep = segmented_trajs[idx]
        nsample = len(ep['seg_ps'])
        ep_data_grp = data_grp.create_group("demo_{}".format(demo_cnt))
        ep_data_grp.attrs["num_samples"] = nsample
        ep_data_grp.create_dataset("states", data=np.array([]))
        initial_state = ep["initial_state"]
        data_env.reset_to(initial_state)

        ep_obs = []
        for i_idx in range(nsample):
            ep_obs.append(data_env.reset_to({"states": ep["seg_states"][i_idx]}))
        obs = {k: np.array([ep_obs[p_idx][k] for p_idx in range(nsample)]) for k in ep_obs[0].keys()}

        primitive_types = np.zeros((nsample, len(skill_names)))
        primitive_labels = np.zeros((nsample, 1))
        primitive_names = []
        args_list = np.zeros((nsample, output_dim))
        arg_masks_list = np.zeros((nsample, output_dim))

        for p_idx in range(nsample):
            p = ep['seg_ps'][p_idx]
            p_id = skill_names.index(p)
            cur_args = ep['seg_args'][p_idx]
            primitive_types[p_idx][p_id] = 1.
            primitive_labels[p_idx][0] = p_id
            primitive_names.append(p)
            args_list[p_idx] = skill_controller.args_to_output(p_name=p, skill_args=cur_args)
            arg_masks_list[p_idx] = skill_controller.get_skill_param_dim(p_name=p)['mask']
        assert len(primitive_names) == nsample
        total_samples += nsample

        # Augmentation for policy training data
        aug_primitive_types = []
        aug_primitive_labels = []
        aug_args_list = []
        aug_arg_masks_list = []
        aug_obs = {k: [] for k in ep_obs[0].keys()}

        ep_original_obs = []
        for i_idx in range(len(ep['original_states'])):
            _original_obs = []
            for j_idx in range(len(ep['original_states'][i_idx])):
                _original_obs.append(data_env.reset_to({"states": ep['original_states'][i_idx][j_idx]}))
            ep_original_obs.append(_original_obs)

        for p_idx in range(nsample):
            aug_max_bound = min(num_augmentation_per_segment, len(ep_original_obs[p_idx])) if 'p_params' in model_type \
                else max(0, min(num_augmentation_per_segment, len(ep_original_obs[p_idx]) - 11))
            for original_ob_idx in range(aug_max_bound):
                _obs = ep_original_obs[p_idx][original_ob_idx].copy()
                for k in ep_obs[0].keys():
                    aug_obs[k].append(_obs[k])
                aug_primitive_types.append(primitive_types[p_idx])
                aug_primitive_labels.append(primitive_labels[p_idx])
                _aug_args = get_pred_params(
                    idm_params_model=idm_params_model,
                    start_obs=eval_env.reset_to({"states": ep["original_states"][p_idx][original_ob_idx]}),
                    end_obs=eval_env.reset_to({"states": ep["original_states"][p_idx][-1]}),
                    p_type=primitive_names[p_idx], skill_controller=skill_controller
                ) if 'p_params' in model_type else ep['seg_args'][p_idx]
                pred_output = skill_controller.args_to_output(primitive_names[p_idx], _aug_args)
                aug_args_list.append(pred_output)
                aug_arg_masks_list.append(skill_controller.get_skill_param_dim(primitive_names[p_idx])['mask'])

        total_samples += len(aug_args_list)
        ep_data_grp.attrs["num_samples"] += len(aug_args_list)
        if 'p_params' in model_type:
            ep_data_grp.create_dataset("actions", data=np.concatenate([args_list, np.array(aug_args_list)]))
            ep_data_grp.create_dataset("action_masks", data=np.concatenate([arg_masks_list, np.array(aug_arg_masks_list)]))
        elif 'p_type' in model_type:
            ep_data_grp.create_dataset("actions", data=np.concatenate([primitive_types, np.array(aug_primitive_types)]))
        ep_data_grp.create_dataset("obs/primitive_type",  data=np.concatenate([primitive_types, np.array(aug_primitive_types)]))
        ep_data_grp.create_dataset("obs/primitive_label", data=np.concatenate([primitive_labels, np.array(aug_primitive_labels)]))
        for k in ep_obs[0]:
            ep_data_grp.create_dataset("obs/{}".format(k), data=np.concatenate([obs[k], np.array(aug_obs[k])]))
        demo_cnt += 1
    data_grp.attrs["total"] = total_samples
    print("Total samples", total_samples)
    split_train_val_from_hdf5(hdf5_path=output_path, val_ratio=val_ratio)