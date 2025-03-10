import numpy as np
from PIL import Image, ImageDraw
from tqdm import trange

def map_segments_to_p_types(idm_type_model, segments, skill_names=None):
    batch_size = 1400
    predictions, probs = [], []
    for batch_idx in trange(int(np.ceil(len(segments) / batch_size))):
        batch = segments[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        cur_batch_size = len(batch)

        # Prepare batch observations
        batch_obs = {
            f"{prefix}_{key}": np.zeros((cur_batch_size, *batch[0]['obs_pair'][0][key].shape))
            for key in batch[0]['obs_pair'][0]
            for prefix in ['cur', 'goal']
        }
        for seq_idx, seq in enumerate(batch):
            for key in seq['obs_pair'][0]:
                batch_obs["cur_{}".format(key)][seq_idx] = seq['obs_pair'][0][key]
                batch_obs["goal_{}".format(key)][seq_idx] = seq['obs_pair'][-1][key]

        # Get predictions and probabilities
        cur_predictions, cur_probs = idm_type_model(ob=batch_obs, ret_prob=True, batch_input=True)
        predictions.extend(cur_predictions.squeeze(-1).astype(int))
        probs.extend(cur_probs.squeeze(-1))

    return [
        dict(
            p=skill_names[pred],
            args=None,  # Primitive parameters will be filled in later using idm_params_model
            prob=prob if pred not in [skill_names.index('atomic')] else (0.0001 * prob),  # Penalize atomic actions
            l=seg['l'],
            r=seg['r'],
            obs_pair=seg['obs_pair'],
            state_pair=seg['state_pair'],
        )
        for seg, pred, prob in zip(segments, predictions, probs)
    ]

def get_pred_params(idm_params_model, start_obs, end_obs, p_type, skill_controller):
    input_obs = dict()
    for key in start_obs:
        input_obs["cur_{}".format(key)] = start_obs[key]
        input_obs["goal_{}".format(key)] = end_obs[key]
    skill_names = skill_controller.primitive_set
    p_id = skill_names.index(p_type)
    p_onehot = np.zeros(len(skill_names))
    p_onehot[p_id] = 1.
    input_obs["primitive_type"] = p_onehot
    output = idm_params_model(ob=input_obs)
    action = skill_controller.output_to_args(p_type, output)
    return action


def parse_demo_to_primitive_seq(traj, idm_type_model, idm_params_model, max_primitive_horizon, algo, skill_controller,
                                verbose):
    skill_names = skill_controller.primitive_set
    idm_type_model.start_episode()
    idm_params_model.start_episode()
    obs_list, action_list, state_list = traj
    assert len(obs_list) == len(state_list) == len(action_list) + 1
    min_primitive_horizon = 5

    segments = []
    for i in range(len(obs_list)):
        for j in range(max(i-max_primitive_horizon, 0), i-min_primitive_horizon):
            segments.append(dict(obs_pair=[obs_list[j], obs_list[i]], l=j, r=i, state_pair=[state_list[j], state_list[i]]))

    p_type_seqs = map_segments_to_p_types(idm_type_model=idm_type_model, segments=segments, skill_names=skill_names)
    dp_func = np.ones(len(obs_list)) * -1e5  # -inf
    seq_idx = 0
    dp_p_seqs = [[] for _ in range(len(obs_list))]
    for i in trange(len(obs_list)):
        for j in range(max(i-max_primitive_horizon, 0), i-min_primitive_horizon):
            cur_primitive = p_type_seqs[seq_idx]
            assert cur_primitive['l'] == j and cur_primitive['r'] == i
            log_prob = np.log(p_type_seqs[seq_idx]['prob'])
            if j == 0:
                if dp_func[i] < log_prob:
                    dp_func[i] = log_prob
                    dp_p_seqs[i] = [cur_primitive]
            else:
                if dp_func[i] < dp_func[j] + log_prob:
                    dp_func[i] = dp_func[j] + log_prob
                    dp_p_seqs[i] = [cur_primitive]
            seq_idx += 1

    # Backtrack to get the primitive sequence
    backtrack_idx = len(obs_list) - 1
    segmented_p_seq = []
    while backtrack_idx > 0:
        backtrack_idx = int(backtrack_idx)
        curr_seq = dp_p_seqs[backtrack_idx][0]
        if curr_seq['p'] != 'atomic':
            segmented_p_seq.append(curr_seq)
        backtrack_idx = curr_seq['l']
    segmented_p_seq.reverse()

    # Construct the segmented trajectory
    segmented_traj = dict(seg_obs=[], seg_states=[], seg_ps=[], seg_args=[], original_states=[])
    prev_pointer = 0
    for _p in segmented_p_seq:
        for j in range(prev_pointer, _p['l']):
            segmented_traj['seg_obs'].append(obs_list[j])
            segmented_traj['seg_states'].append(state_list[j])
            segmented_traj['seg_ps'].append('atomic')
            segmented_traj['seg_args'].append(action_list[j])
            segmented_traj['original_states'].append(state_list[j:j+2])
        _p['args'] = get_pred_params(idm_params_model, _p['obs_pair'][0], _p['obs_pair'][-1], _p['p'],
                                     skill_controller)
        if verbose:
            print('{}: [{}, {}], prob: {}'.format(_p['p'], _p['l'], _p['r'], _p['prob']))
        segmented_traj['seg_obs'].append(_p['obs_pair'][0])
        segmented_traj['seg_states'].append(_p['state_pair'][0])
        segmented_traj['seg_ps'].append(_p['p'])
        segmented_traj['seg_args'].append(_p['args'])
        segmented_traj['original_states'].append(state_list[_p['l']:_p['r'] + 1])
        prev_pointer = _p['r']

    for j in range(prev_pointer, len(obs_list) - 1):
        segmented_traj['seg_obs'].append(obs_list[j])
        segmented_traj['seg_states'].append(state_list[j])
        segmented_traj['seg_ps'].append('atomic')
        segmented_traj['seg_args'].append(action_list[j])
        segmented_traj['original_states'].append(state_list[j:j+2])

    segmented_traj['seg_obs'].append(obs_list[-1])
    segmented_traj['seg_states'].append(state_list[-1])
    assert sum([len(_state) - 1 for _state in segmented_traj['original_states']]) == len(obs_list) - 1
    return segmented_traj, dp_func[-1]


def test_with_primitives(eval_env, initial_state, segmented_traj, skill_controller):
    eval_env.reset_to(initial_state)
    for p_idx in range(len(segmented_traj['seg_ps'])):
        if segmented_traj['seg_ps'][p_idx] != 'atomic':
            skill_controller.execute(p_name=segmented_traj['seg_ps'][p_idx], skill_args=segmented_traj['seg_args'][p_idx], norm=True)
    return eval_env.env._check_success()
