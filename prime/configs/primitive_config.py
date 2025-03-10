def get_primitive_config(env_name):
    if env_name in ['StackTwoTypesDomain', 'SortTwoObjectsDomain', 'NutAssemblySquare', 'NutAssembly',
                    'NutAssemblyRound', 'NutAssemblyRoundSmallInit', 'PickPlaceCan', 'PickPlace', 'PickPlaceMilk']:
        return ['reach', 'place', 'grasp', 'atomic']
    elif env_name in ['CleanUpMediumSmallInitD1', 'CleanUpMediumSmallInitD2']:
        return ['reach', 'place', 'grasp', 'push', 'atomic']
    else:
        raise NotImplementedError