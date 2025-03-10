import sys
sys.path.append('.')
import imageio
import numpy as np
from PIL import Image, ImageDraw
import os
from tqdm import trange

from prime.configs.input_args import get_args
from prime.configs.primitive_config import get_primitive_config

from robosuite.controllers.skill_controller import SkillController
import robosuite.utils.macros as macros

import robomimic.utils.file_utils as FileUtils

# macros.IMAGE_CONVENTION = "opencv"

def eval_policy(policy_type_path, policy_params_path, env_horizon, device, num_rollouts, render, write_video,
                 output_mode, controller, primitive_set, verbose):
    policy_type_model, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=policy_type_path, device=device, verbose=False)
    policy_type_model.start_episode()
    policy_params_model, ckpt = FileUtils.policy_from_checkpoint(ckpt_path=policy_params_path, device=device, verbose=False)
    policy_params_model.start_episode()

    env_name = ckpt_dict["env_metadata"]["env_name"]
    eval_env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=env_name,
        render=render,
        render_offscreen=write_video,
        verbose=False,
    )
    primitives_kwargs = dict(
        render=render,
        controller_type=controller,
        image_obs_in_info=write_video,
        output_mode=output_mode,
        primitive_set=get_primitive_config(env_name) if primitive_set is None else primitive_set
    )
    skill_controller = SkillController(env=eval_env, **primitives_kwargs)
    skill_names = skill_controller.primitive_set

    total_rollouts = successful_rollouts = 0
    video_images = []

    eval_env.reset()
    state_dict = eval_env.get_state()
    obs = eval_env.reset_to(state_dict)
    if write_video:
        video_images.append(eval_env.render(mode="rgb_array", height=256, width=256, camera_name='agentview'))
    primitive_nsteps_list, atomic_nactions_list = [], []
    ep_idx, ep_primitive_nsteps, ep_atomic_nactions = 0, 0, 0
    while ep_idx < num_rollouts:
        pred_type, _ = policy_type_model(ob=obs, ret_prob=True)
        pred_type = pred_type.squeeze()
        obs['primitive_type'] = np.eye(len(skill_controller.primitive_set))[pred_type]
        pred_params = policy_params_model(ob=obs)
        pred_args = skill_controller.output_to_args(skill_names[pred_type], pred_params)
        ret = skill_controller.execute(p_name=skill_names[pred_type], skill_args=pred_args, norm=True)

        ep_primitive_nsteps += 1
        ep_atomic_nactions += ret['info']['num_ac_calls']
        if verbose:
            print("Primitive {}: {}".format(ep_primitive_nsteps, skill_names[pred_type]))

        obs = eval_env.get_observation()
        if write_video:
            video_images.extend([np.array(Image.fromarray(img, 'RGB')) for img in ret['info']['image_obs']])
        if eval_env.is_success()["task"] or ep_atomic_nactions >= env_horizon:
            total_rollouts += 1
            if eval_env.is_success()["task"]:
                successful_rollouts += 1
                if verbose:
                    print("Rollout {} successful! Primitive steps: {}, Atomic actions: {}".format(ep_idx, ep_primitive_nsteps, ep_atomic_nactions))
            ep_idx += 1
            primitive_nsteps_list.append(ep_primitive_nsteps)
            atomic_nactions_list.append(ep_atomic_nactions)
            ep_primitive_nsteps, ep_atomic_nactions = 0, 0
            policy_type_model.start_episode()
            policy_params_model.start_episode()
            eval_env.reset()
            state_dict = eval_env.get_state()
            obs = eval_env.reset_to(state_dict)
            if write_video:
                video_images.append(eval_env.render(mode="rgb_array", height=256, width=256, camera_name='agentview'))
    if write_video:
        os.makedirs('policy_results/{}'.format(env_name), exist_ok=True)
        video_writer = imageio.get_writer('policy_results/{}/policy_rollout.mp4'.format(env_name), fps=5)
        for i_img in range(0, len(video_images), 10):
            video_writer.append_data(video_images[i_img])
        video_writer.close()
    sr = successful_rollouts / total_rollouts
    print("Policy success rate: {:.2f}%".format(sr * 100))
    print("Mean primitive steps: {:.2f}".format(np.mean(primitive_nsteps_list)))
    print("Mean atomic actions: {:.2f}".format(np.mean(atomic_nactions_list)))
    return sr

if __name__ == '__main__':
    args = get_args()
    print("Evaluating policy...")
    success_rates = []
    for i in range(1, 11):
        policy_type_model_path = os.path.join(args.policy_type_model_dir, "model_epoch_{}.pth".format(i))
        policy_params_model_path = os.path.join(args.policy_params_model_dir, "model_epoch_{}.pth".format(i))
        print("="*20)
        print("Evaluating policy at epoch {}".format(i))
        sr = eval_policy(policy_type_model_path, policy_params_model_path, env_horizon=args.env_horizon,
                    device=args.device, num_rollouts=args.num_rollouts,
                    render=args.render, write_video=args.write_video, output_mode=args.output_mode,
                    controller=args.controller, primitive_set=args.primitive_set, verbose=args.verbose)
        success_rates.append(sr)

    print("Policy evaluation complete!")
    print("Top 5 success rates: {}".format(sorted(success_rates, reverse=True)[:5], reversed=True))