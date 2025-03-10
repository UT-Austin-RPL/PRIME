import random
import numpy as np
import os
from tqdm import tqdm
import fcntl
import pickle
import logging

from prime.configs.primitive_config import get_primitive_config

from robosuite import load_controller_config
from robosuite.controllers.skill_controller import SkillController

import robomimic.envs.env_base as EB
import robomimic.utils.env_utils as EnvUtils

def safe_div(a, b):
    if b == 0:
        return 0
    else:
        return a / b

def setup_logger(log_file):
    """Setup a logger for a specific process."""
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

class TqdmToLogger:
    """Redirects tqdm output to the logging file."""
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, message):
        # Avoid writing empty lines
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        """Required for compatibility with file-like objects."""
        pass

def get_valid_primitives(primitive_lst, skill_controller):
    valid_primitives = []
    for p in primitive_lst:
        if skill_controller.test_start_state(p):
            valid_primitives.append(p)
    return valid_primitives

def collect_primitive_rollouts(params):
    num_trajs, env_name, num_primitives, render, data_dir, save, seed, controller, output_mode, primitive_set,\
         verbose = [params[key] for key in ['num_trajs', 'env_name', 'num_primitives', 'render', 'data_dir', 'save',
                                            'seed', 'controller', 'output_mode', 'primitive_set', 'verbose']]
    random.seed(seed)
    np.random.seed(seed)
    data_path = os.path.join(data_dir, env_name)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(os.path.join(data_path, 'data_logs'), exist_ok=True)
    log_file = os.path.join(os.path.join(data_path, 'data_logs'), 'data_{}.log'.format(seed))
    logger = setup_logger(log_file)
    tqdm_logger = TqdmToLogger(logger)
    if verbose:
        print("Logging to {}".format(log_file))
    if save:
        data_file = os.path.join(data_path, 'data_{}.pkl'.format(seed))
        assert not os.path.exists(data_file)
        if verbose:
            print("Saving rollouts in {}".format(data_file))
        logger.info("Saving rollouts in {}".format(data_file))

    controller_name = controller
    controller_config = load_controller_config(default_controller=controller_name)
    env_config = {
        "env_name": env_name,
        "robots": "Panda",
        "controller_configs": controller_config,
    }
    env_kwargs = dict(
        **env_config,
        has_renderer=render,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        horizon=5000,
    )
    env_meta = dict(
        type=EB.EnvType.ROBOSUITE_TYPE,
        env_name=env_name,
        env_kwargs=env_kwargs,
    )
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=[],
        camera_height=84,
        camera_width=84,
        reward_shaping=False,
    )

    if save:
        env_file_path = os.path.join(data_path, 'env_meta.pkl')
        try:
            with open(env_file_path, "xb") as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                pickle.dump(dict(env_meta=env_meta), f)
                logger.info(f"File {env_file_path} written successfully.")
        except FileExistsError:
            logger.error(f"File {env_file_path} already exists. Skipping write.")
        except BlockingIOError:
            logger.error(f"File {env_file_path} is locked by another process. Skipping write.")
        except Exception as e:
            logger.error(f"Error writing file {env_file_path}: {e}")

    env.reset()
    skill_controller = SkillController(env=env,
                                       controller_type=controller_name,
                                       image_obs_in_info=False,
                                       aff_type=None,
                                       render=render,
                                       primitive_set=get_primitive_config(env_name) if (primitive_set is None) else primitive_set,
                                       output_mode=output_mode,
                                       )
    primitive_lst = list(skill_controller.name_to_skill.keys())
    skill_ntraj = {p_name: 0 for p_name in primitive_lst}
    skill_nstep = {p_name: 0 for p_name in primitive_lst}
    interest_ntraj = {p_name: 0 for p_name in primitive_lst}
    saved_interest_ntraj = {p_name: 0 for p_name in primitive_lst}
    skill_ntraj['total'] = skill_nstep['total'] = 0

    ep_cnt = 0
    for ep_idx in tqdm(range(num_trajs), file=tqdm_logger, desc='Collecting episodes'):
        concatenated_traj_list = dict(state=[], action=[], primitive=[], initial_state=None, tag='combine')
        env.reset()
        task_init_state = env.get_state()
        obs = env.get_observation()
        state = env.get_state()
        concatenated_traj_list['state'].append(state)
        concatenated_traj_list['initial_state'] = task_init_state
        save_concatenated_traj = False
        for idx in tqdm(range(num_primitives), file=tqdm_logger, desc='Rolling out primitives'):
            primitive_task_init_state = env.get_state()
            valid_primitive_lst = get_valid_primitives(primitive_lst, skill_controller)
            p_name = random.choices(valid_primitive_lst, k=1)[0]
            primitive = skill_controller.get_skill(p_name=p_name)
            low, high = primitive.get_param_spec()
            logger.info("ep id {}, p id {}: {}".format(ep_idx, idx, p_name))
            if p_name == 'atomic':
                nstep = random.randint(20, 60)
                state_list = [primitive_task_init_state["states"]]
                action_list = []
                for _astep in range(nstep):
                    action = np.random.uniform(low, high)
                    ret = skill_controller.execute(p_name=p_name, skill_args=action, norm=False)
                    assert len(ret['info']['state_list']) == 2
                    state_list += ret['info']['state_list'][1:]
                    action_list += ret['info']['action_list']
                    concatenated_traj_list['primitive'].append(dict(p_name=p_name,
                                                               args=action,
                                                               action_start_idx=len(concatenated_traj_list['action']) + _astep,
                                                               p_len=1,
                                                               interest=ret['info']['interest_interaction'],
                                                               done=ret['info']['done_interaction']))
            else:
                aff_centers = primitive.get_aff_centers()
                aff_noise_scale = primitive.get_aff_noise_scale()
                if aff_centers is not None:
                    if len(aff_centers) == 0:
                        continue
                    for _ in tqdm(range(10), file=tqdm_logger, desc="Searching for valid target position"):  # Try 10 times to sample a valid target position
                        env.reset_to({"states": primitive_task_init_state["states"]})
                        aff_center = random.choices(aff_centers, k=1)[0]
                        action = np.random.uniform(low, high)
                        target_pos = aff_center + np.random.randn(3) * aff_noise_scale # Add noise to the target position
                        action[:3] = primitive._get_normalized_params(target_pos, bounds=primitive._config['global_xyz_bounds'])
                        ret = skill_controller.execute(p_name, skill_args=action, norm=True)
                        if ret['info']['interest_interaction']:
                            break
                else:
                    action = np.random.uniform(low, high)
                    ret = skill_controller.execute(p_name, skill_args=action, norm=True)

                state_list = ret['info']['state_list']
                action_list = ret['info']['action_list']
                nstep = ret['info']['num_ac_calls']
                concatenated_traj_list['primitive'].append(dict(p_name=p_name, args=action,
                                                           action_start_idx=len(concatenated_traj_list['action']),
                                                           p_len=nstep,
                                                           interest=ret['info']['interest_interaction'],
                                                           done=ret['info']['done_interaction']))

                p_traj = dict(state=state_list, action=action_list,
                              initial_state=primitive_task_init_state,
                              tag=p_name, args=action,
                              interest=ret['info']['interest_interaction'],
                              done=ret['info']['done_interaction'])
                if ret['info']['interest_interaction'] and save:
                    saved_interest_ntraj[p_name] += 1
                    with open(data_file, "ab") as f:
                        pickle.dump(p_traj, f)
            assert len(state_list) - 1 == len(action_list) == nstep
            concatenated_traj_list['state'] += state_list[1:]
            concatenated_traj_list['action'] += action_list
            skill_nstep[p_name] += nstep
            skill_ntraj[p_name] += 1
            skill_ntraj['total'] += 1
            skill_nstep['total'] += nstep
            if ret['info']['interest_interaction']:
                interest_ntraj[p_name] += 1
            if ep_cnt < num_trajs:
                save_concatenated_traj = True
        logger.info(f"episode {ep_cnt + 1}")
        logger.info(f"total nsteps {skill_nstep['total']}, total ntraj {skill_ntraj['total']}")
        for _p_name in primitive_lst:
            logger.info(f"{_p_name} nstep: {skill_nstep[_p_name]}, {_p_name} ntraj: {skill_ntraj[_p_name]}, {_p_name} interest ntraj: {interest_ntraj[_p_name]}, {_p_name} saved interest ntraj: {saved_interest_ntraj[_p_name]}, interest rate: {safe_div(interest_ntraj[_p_name], skill_ntraj[_p_name])}")
        if save and save_concatenated_traj:
            logger.info("save concatenated traj {}".format(ep_cnt + 1))
            with open(data_file, "ab") as f:
                pickle.dump(concatenated_traj_list, f)
        ep_cnt += 1
    if save:
        logger.info("Finished collecting data in file {}".format(data_file))
        if verbose:
            print("Finished saving data in file: ", data_file)
