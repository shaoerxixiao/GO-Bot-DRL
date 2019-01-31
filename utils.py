from dialogue_config import FAIL, SUCCESS


def convert_list_to_dict(lst):
    """
    将list转化为dict, keys为list的值，values为标号

    参数:
        lst (list)

    返回:
        dict
    """

    if len(lst) > len(set(lst)):
        raise ValueError('List must be unique!')
    return {k: v for v, k in enumerate(lst)}


def remove_empty_slots(dic):
    """
    Removes all items with values of '' (ie values of empty string).

    Parameters:
        dic (dict)
    """

    for id in list(dic.keys()):
        for key in list(dic[id].keys()):
            if dic[id][key] == '':
                dic[id].pop(key)


def reward_function(success, max_round):
    """
    Return the reward given the success.

    Return -1 + -max_round if success is FAIL, -1 + 2 * max_round if success is SUCCESS and -1 otherwise.

    Parameters:
        success (int)

    Returns:
        int: Reward
    """

    reward = -1
    if success == FAIL:
        reward += -max_round
    elif success == SUCCESS:
        reward += 2 * max_round
    return reward
