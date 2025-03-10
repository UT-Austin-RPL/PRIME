import sys
import os
sys.path.append('.')
from multiprocessing import Pool

from prime.configs.input_args import get_args
from prime.core.collect_data import collect_primitive_rollouts
from prime.core.segment_data import segment_demos
from prime.utils.data_utils import reformat_rollout_data, reformat_segmented_trajs


if __name__ == "__main__":
    args = get_args()
    print("Input args", args)
    if args.collect_demos:
        print("Collecting demos...")
        with Pool(processes=args.num_data_workers) as pool:
            pool.map(collect_primitive_rollouts, [
                dict(
                    num_trajs=args.num_trajs, env_name=args.env, render=args.render, data_dir=args.data_dir,
                    save=args.save, seed=seed, num_primitives=args.num_primitives, controller=args.controller,
                    output_mode=args.output_mode, primitive_set=args.primitive_set, verbose=args.verbose
                )
                for seed in range(args.num_data_workers)
            ])
        print("Finish collecting demos!")

    if args.reformat_rollout_data:
        print("Reformatting demos into hdf5 file...")
        reformat_rollout_data(args.data_dir, args.output_mode, args.controller, args.env,
                                           args.primitive_set, 'idm_p_type', args.num_data_workers, args.num_others_per_traj,
                                           args.val_ratio, args.camera_names, args.camera_height, args.camera_width)
        reformat_rollout_data(args.data_dir, args.output_mode, args.controller, args.env,
                                           args.primitive_set, 'idm_p_params', args.num_data_workers, args.num_others_per_traj,
                                           args.val_ratio, args.camera_names, args.camera_height, args.camera_width)
        if args.policy_pretrain:
            reformat_rollout_data(args.data_dir, args.output_mode, args.controller, args.env,
                                   args.primitive_set, 'policy_pt_p_params', args.num_data_workers, args.num_others_per_traj,
                                   args.val_ratio, args.camera_names, args.camera_height, args.camera_width)
        print("Finish reformatting demos!")


    if args.segment_demos:
        print("Segmenting demos...")
        segmented_data = segment_demos(demo_path=args.demo_path, num_demos=args.num_demos, device=args.device, render=args.render,
                                       idm_type_model_path=args.idm_type_model_path, idm_params_model_path=args.idm_params_model_path,
                                       primitive_set=args.primitive_set, output_mode=args.output_mode, controller=args.controller,
                                       segmented_data_dir=args.segmented_data_dir, save_failed_trajs=args.save_failed_trajs,
                                       parser_algo=args.parser_algo, max_primitive_horizon=args.max_primitive_horizon,
                                       verbose=args.verbose, playback_segmented_trajs=args.playback_segmented_trajs,)
        print("Finish segmenting demos!")

        print("Reformatting segmented demos...")
        reformat_segmented_trajs(segmented_data, args.segmented_data_dir, args.num_augmentation_type,
                                 args.idm_params_model_path, model_type='policy_p_type',
                                 controller=args.controller, output_mode=args.output_mode,
                                 primitive_set=args.primitive_set,
                                 device=args.device, val_ratio=args.val_ratio,
                                 camera_names=args.camera_names, camera_height=args.camera_height,
                                 camera_width=args.camera_width)
        reformat_segmented_trajs(segmented_data, args.segmented_data_dir, args.num_augmentation_params,
                                 args.idm_params_model_path, model_type='policy_p_params',
                                 controller=args.controller, output_mode=args.output_mode,
                                 primitive_set=args.primitive_set,
                                 device=args.device, val_ratio=args.val_ratio,
                                 camera_names=args.camera_names, camera_height=args.camera_height,
                                 camera_width=args.camera_width)
        print("Finish reformatting segmented demos!")

