import numpy as np
from copy import deepcopy

import robosuite as suite
import robosuite.utils.transform_utils as T

import robomimic.envs.env_base as EB

def get_env_class(env_meta=None, env_type=None, env=None):
    """
    Return env class from either env_meta, env_type, or env.
    Note the use of lazy imports - this ensures that modules are only
    imported when the corresponding env type is requested. This can
    be useful in practice. For example, a training run that only
    requires access to gym environments should not need to import
    robosuite.

    Args:
        env_meta (dict): environment metadata, which should be loaded from demonstration
            hdf5 with @FileUtils.get_env_metadata_from_dataset or from checkpoint (see
            @FileUtils.env_from_checkpoint). Contains 3 keys:

                :`'env_name'`: name of environment
                :`'type'`: type of environment, should be a value in EB.EnvType
                :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor

        env_type (int): the type of environment, which determines the env class that will
            be instantiated. Should be a value in EB.EnvType.

        env (instance of EB.EnvBase): environment instance
    """
    env_type = get_env_type(env_meta=env_meta, env_type=env_type, env=env)
    if env_type == EB.EnvType.ROBOSUITE_TYPE:
        from prime.envs.env_robosuite import EnvRobosuite
        return EnvRobosuite
    elif env_type == EB.EnvType.GYM_TYPE:
        from robomimic.envs.env_gym import EnvGym
        return EnvGym
    elif env_type == EB.EnvType.IG_MOMART_TYPE:
        from robomimic.envs.env_ig_momart import EnvGibsonMOMART
        return EnvGibsonMOMART
    raise Exception("code should never reach this point")


def get_env_type(env_meta=None, env_type=None, env=None):
    """
    Helper function to get env_type from a variety of inputs.

    Args:
        env_meta (dict): environment metadata, which should be loaded from demonstration
            hdf5 with @FileUtils.get_env_metadata_from_dataset or from checkpoint (see
            @FileUtils.env_from_checkpoint). Contains 3 keys:

                :`'env_name'`: name of environment
                :`'type'`: type of environment, should be a value in EB.EnvType
                :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor

        env_type (int): the type of environment, which determines the env class that will
            be instantiated. Should be a value in EB.EnvType.

        env (instance of EB.EnvBase): environment instance
    """
    checks = [(env_meta is not None), (env_type is not None), (env is not None)]
    assert sum(checks) == 1, "should provide only one of env_meta, env_type, env"
    if env_meta is not None:
        env_type = env_meta["type"]
    elif env is not None:
        env_type = env.type
    return env_type


def check_env_type(type_to_check, env_meta=None, env_type=None, env=None):
    """
    Checks whether the passed env_meta, env_type, or env is of type @type_to_check.
    Type corresponds to EB.EnvType.

    Args:
        type_to_check (int): type to check equality against

        env_meta (dict): environment metadata, which should be loaded from demonstration
            hdf5 with @FileUtils.get_env_metadata_from_dataset or from checkpoint (see
            @FileUtils.env_from_checkpoint). Contains 3 keys:

                :`'env_name'`: name of environment
                :`'type'`: type of environment, should be a value in EB.EnvType
                :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor

        env_type (int): the type of environment, which determines the env class that will
            be instantiated. Should be a value in EB.EnvType.

        env (instance of EB.EnvBase): environment instance
    """
    env_type = get_env_type(env_meta=env_meta, env_type=env_type, env=env)
    return (env_type == type_to_check)


def is_robosuite_env(env_meta=None, env_type=None, env=None):
    """
    Determines whether the environment is a robosuite environment. Accepts
    either env_meta, env_type, or env.
    """
    return check_env_type(type_to_check=EB.EnvType.ROBOSUITE_TYPE, env_meta=env_meta, env_type=env_type, env=env)


def create_env(
    env_type,
    env_name,
    render=False,
    render_offscreen=False,
    use_image_obs=False,
    **kwargs,
):
    """
    Create environment.

    Args:
        env_type (int): the type of environment, which determines the env class that will
            be instantiated. Should be a value in EB.EnvType.

        env_name (str): name of environment

        render (bool): if True, environment supports on-screen rendering

        render_offscreen (bool): if True, environment supports off-screen rendering. This
            is forced to be True if @use_image_obs is True.

        use_image_obs (bool): if True, environment is expected to render rgb image observations
            on every env.step call. Set this to False for efficiency reasons, if image
            observations are not required.
    """

    # note: pass @postprocess_visual_obs True, to make sure images are processed for network inputs
    env_class = get_env_class(env_type=env_type)
    env = env_class(
        env_name=env_name,
        render=render,
        render_offscreen=render_offscreen,
        use_image_obs=use_image_obs,
        postprocess_visual_obs=True,
        **kwargs,
    )
    print("Created environment with name {}".format(env_name))
    return env


def create_env_from_metadata(
    env_meta,
    env_name=None,
    render=False,
    render_offscreen=False,
    use_image_obs=False,
):
    """
    Create environment.

    Args:
        env_meta (dict): environment metadata, which should be loaded from demonstration
            hdf5 with @FileUtils.get_env_metadata_from_dataset or from checkpoint (see
            @FileUtils.env_from_checkpoint). Contains 3 keys:

                :`'env_name'`: name of environment
                :`'type'`: type of environment, should be a value in EB.EnvType
                :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor

        env_name (str): name of environment. Only needs to be provided if making a different
            environment from the one in @env_meta.

        render (bool): if True, environment supports on-screen rendering

        render_offscreen (bool): if True, environment supports off-screen rendering. This
            is forced to be True if @use_image_obs is True.

        use_image_obs (bool): if True, environment is expected to render rgb image observations
            on every env.step call. Set this to False for efficiency reasons, if image
            observations are not required.
    """
    if env_name is None:
        env_name = env_meta["env_name"]
    env_type = get_env_type(env_meta=env_meta)
    env_kwargs = env_meta["env_kwargs"]

    env = create_env(
        env_type=env_type,
        env_name=env_name,
        render=render,
        render_offscreen=render_offscreen,
        use_image_obs=use_image_obs,
        **env_kwargs,
    )
    return env


def create_env_for_data_processing(
    env_meta,
    camera_names,
    camera_height,
    camera_width,
    reward_shaping,
):
    """
    Creates environment for processing dataset observations and rewards.

    Args:
        env_meta (dict): environment metadata, which should be loaded from demonstration
            hdf5 with @FileUtils.get_env_metadata_from_dataset or from checkpoint (see
            @FileUtils.env_from_checkpoint). Contains 3 keys:

                :`'env_name'`: name of environment
                :`'type'`: type of environment, should be a value in EB.EnvType
                :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor

        camera_names (list of st): list of camera names that correspond to image observations

        camera_height (int): camera height for all cameras

        camera_width (int): camera width for all cameras

        reward_shaping (bool): if True, use shaped environment rewards, else use sparse task completion rewards
    """
    env_name = env_meta["env_name"]
    env_type = get_env_type(env_meta=env_meta)
    env_kwargs = env_meta["env_kwargs"]
    env_class = get_env_class(env_type=env_type)

    # remove possibly redundant values in kwargs
    env_kwargs = deepcopy(env_kwargs)
    env_kwargs.pop("env_name", None)
    env_kwargs.pop("camera_names", None)
    env_kwargs.pop("camera_height", None)
    env_kwargs.pop("camera_width", None)
    env_kwargs.pop("reward_shaping", None)

    return env_class.create_for_data_processing(
        env_name=env_name,
        camera_names=camera_names,
        camera_height=camera_height,
        camera_width=camera_width,
        reward_shaping=reward_shaping,
        **env_kwargs,
    )

def create_env_for_segmentation_evaluation(
    env_meta,
    camera_names,
    camera_height,
    camera_width,
    reward_shaping,
):
    """
    Creates environment for processing dataset observations and rewards.

    Args:
        env_meta (dict): environment metadata, which should be loaded from demonstration
            hdf5 with @FileUtils.get_env_metadata_from_dataset or from checkpoint (see
            @FileUtils.env_from_checkpoint). Contains 3 keys:

                :`'env_name'`: name of environment
                :`'type'`: type of environment, should be a value in EB.EnvType
                :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor

        camera_names (list of st): list of camera names that correspond to image observations

        camera_height (int): camera height for all cameras

        camera_width (int): camera width for all cameras

        reward_shaping (bool): if True, use shaped environment rewards, else use sparse task completion rewards
    """
    env_name = env_meta["env_name"]
    env_type = get_env_type(env_meta=env_meta)
    env_kwargs = env_meta["env_kwargs"]
    env_class = get_env_class(env_type=env_type)

    # remove possibly redundant values in kwargs
    env_kwargs = deepcopy(env_kwargs)
    env_kwargs.pop("env_name", None)
    env_kwargs.pop("camera_names", None)
    env_kwargs.pop("camera_height", None)
    env_kwargs.pop("camera_width", None)
    env_kwargs.pop("reward_shaping", None)

    return env_class.create_for_segmentation_evaluation(
        env_name=env_name,
        camera_names=camera_names,
        camera_height=camera_height,
        camera_width=camera_width,
        reward_shaping=reward_shaping,
        **env_kwargs,
    )

def create_env_for_policy_evaluation(
    env_meta,
    camera_names,
    camera_height,
    camera_width,
    reward_shaping,
):
    """
    Creates environment for processing dataset observations and rewards.

    Args:
        env_meta (dict): environment metadata, which should be loaded from demonstration
            hdf5 with @FileUtils.get_env_metadata_from_dataset or from checkpoint (see
            @FileUtils.env_from_checkpoint). Contains 3 keys:

                :`'env_name'`: name of environment
                :`'type'`: type of environment, should be a value in EB.EnvType
                :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor

        camera_names (list of st): list of camera names that correspond to image observations

        camera_height (int): camera height for all cameras

        camera_width (int): camera width for all cameras

        reward_shaping (bool): if True, use shaped environment rewards, else use sparse task completion rewards
    """
    env_name = env_meta["env_name"]
    env_type = get_env_type(env_meta=env_meta)
    env_kwargs = env_meta["env_kwargs"]
    env_class = get_env_class(env_type=env_type)

    # remove possibly redundant values in kwargs
    env_kwargs = deepcopy(env_kwargs)
    env_kwargs.pop("env_name", None)
    env_kwargs.pop("camera_names", None)
    env_kwargs.pop("camera_height", None)
    env_kwargs.pop("camera_width", None)
    env_kwargs.pop("reward_shaping", None)

    return env_class.create_for_policy_evaluation(
        env_name=env_name,
        camera_names=camera_names,
        camera_height=camera_height,
        camera_width=camera_width,
        reward_shaping=reward_shaping,
        **env_kwargs,
    )


def get_eef_pos(obs):
    return obs['robot0_eef_pos']

def get_eef_quat(obs):
    if obs['robot0_eef_quat'][-1] < 0:
        return - obs['robot0_eef_quat']
    return obs['robot0_eef_quat']

def get_gripper_dist(obs):
    return obs['robot0_gripper_qpos'][0] - obs['robot0_gripper_qpos'][1]

def get_obs(env):
    _is_v1 = (suite.__version__.split(".")[0] == "1")
    if _is_v1:
        assert (int(suite.__version__.split(".")[1]) >= 2), "only support robosuite v0.3 and v1.2+"
    obs = (
        env._get_observations(force_update=True)
        if _is_v1
        else env._get_observation()
    )
    return obs

def stablize_env(env, render=False, write_video=False, video_kwargs=None):
    for _ in range(4):
        obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape[0]))
        if render:
            env.render()
        if write_video:
            video_kwargs['video_writer'].append_data(obs['agentview' + "_image"])
            video_kwargs['video_count'] += 1

def get_axisangle_error(cur_quat, target_quat):
    cur_orn = T.quat2mat(cur_quat)
    goal_orn = T.quat2mat(target_quat)
    rot_error = goal_orn @ cur_orn.T
    quat_error = T.mat2quat(rot_error)
    axisangle_error = T.quat2axisangle(quat_error)
    return axisangle_error

def get_target_quat(cur_quat, axisangle_error):
    quat_error = T.axisangle2quat(axisangle_error)
    rotation_mat_error = T.quat2mat(quat_error)
    current_orientation = T.quat2mat(cur_quat)
    goal_orientation = np.dot(rotation_mat_error, current_orientation)
    goal_quat = T.mat2quat(goal_orientation)
    return goal_quat