import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # env
    parser.add_argument('--env', type=str, default='CleanUp')
    parser.add_argument('--controller', type=str, default='OSC_POSE')
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument("--camera-names", type=str, nargs='+', default=['agentview', 'robot0_eye_in_hand'])
    parser.add_argument("--camera-height", type=int, default=84)
    parser.add_argument("--camera-width", type=int, default=84)

    # skill controller
    parser.add_argument('--primitive-set', type=str, nargs='+')
    parser.add_argument('--output-mode', type=str, default='max')
    parser.add_argument('--num-data-workers', type=int, default=40)

    # data collection
    parser.add_argument('--collect-demos', action='store_true', default=False)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--num-trajs', type=int)
    parser.add_argument('--num-primitives', type=int, default=50)
    parser.add_argument('--save', action='store_true', default=False)

    # data reformat
    parser.add_argument('--reformat-rollout-data', action='store_true', default=False)
    parser.add_argument("--num-others-per-traj", type=int, default=30)
    parser.add_argument("--val-ratio", type=float, default=0.)
    parser.add_argument('--policy-pretrain', action='store_true', default=False)

    # model
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--idm-type-model-path", type=str, default=None)
    parser.add_argument("--idm-params-model-path", type=str, default=None)

    # trajectory parser
    parser.add_argument('--segment-demos', action='store_true', default=False)
    parser.add_argument("--demo-path", type=str, default=None)
    parser.add_argument("--num-demos", type=int, default=None)
    parser.add_argument("--save-failed-trajs", action='store_true', default=False)
    parser.add_argument("--max-primitive-horizon", default=100, type=int)
    parser.add_argument('--segmented-data-dir', type=str, default=None)
    parser.add_argument('--parser-algo', type=str, default='dp')
    parser.add_argument("--playback-segmented-trajs", action='store_true', default=False)
    parser.add_argument("--num-augmentation-type", default=50, type=int)
    parser.add_argument("--num-augmentation-params", default=100, type=int)

    # policy evaluation
    parser.add_argument('--policy-type-model-dir', type=str, default=None)
    parser.add_argument('--policy-params-model-dir', type=str, default=None)
    parser.add_argument("--env-horizon", default=1000, type=int)
    parser.add_argument("--num-rollouts", default=50, type=int)

    # visualization
    parser.add_argument("--write-video", action='store_true', default=False)
    parser.add_argument("--verbose", action='store_true', default=False)

    args = parser.parse_args()
    return args


