from db_query import DBQuery
import numpy as np
from utils import convert_list_to_dict
from dialogue_config import all_intents, all_slots, usersim_default_key
import copy


class StateTracker:
    """追踪对话的状态，为agent提供当前状态的representation以便让其作出合适的action"""

    def __init__(self, database, constants):
        """
        The constructor of StateTracker.

        The constructor of StateTracker which creates a DB query object, creates necessary state rep. dicts, etc. and
        calls reset.

        Parameters:
            database (dict): The database with format dict(long: dict)
            constants (dict): Loaded constants in dict

        """
        # db查找工具
        self.db_helper = DBQuery(database)
        # 整个对话的目标key，默认为'ticket'
        self.match_key = usersim_default_key
        # intents的dict，key为intent,value为序号
        self.intents_dict = convert_list_to_dict(all_intents)
        # intents个数
        self.num_intents = len(all_intents)
        # slots的dict，key为slot,value为序号
        self.slots_dict = convert_list_to_dict(all_slots)
        # slots个数
        self.num_slots = len(all_slots)
        # 所允许的最长对话回合数，超过此回合则对话失败
        self.max_round_num = constants['run']['max_round_num']
        # 对话状态中的零状态，即什么信息也没有
        self.none_state = np.zeros(self.get_state_size())
        # 初始化StateTracker
        self.reset()

    def get_state_size(self):
        """返回 state representation 的维度"""

        return 2 * self.num_intents + 7 * self.num_slots + 3 + self.max_round_num

    def reset(self):
        """重置StateTracker, 需要初始化current_informs, history and round_num."""

        self.current_informs = {}
        # A list of the dialogues (dicts) by the agent and user so far in the conversation
        self.history = []
        self.round_num = 0

    def print_history(self):
        """查看历史actions"""

        for action in self.history:
            print(action)

    def get_state(self, done=False):
        """
        返回当前的state representation, 表现形式为numpy array，包括user, agent对应的intent, inform_slot, request_slot信息，
        当前状态下已满足条件的slots信息，db的查询结果信息，对话轮次信息

        Parameters:
            done (bool): 表明是否是最后一轮对话，默认为False

        Returns:
            numpy.array: numpy array，形状为 (state size,)

        """

        # 如果为done，则 state 中的值全为0
        if done:
            return self.none_state
        # 取history中的最后一个值，即当前状态下user最近的一个action
        user_action = self.history[-1]
        # 根据current_informs，从db中查询满足条件的信息
        db_results_dict = self.db_helper.get_db_results_for_slots(self.current_informs)
        # 取history中倒数第二个值，即当前状态下agent最近的一个action，如果history的长度小于等于1，则为None
        last_agent_action = self.history[-2] if len(self.history) > 1 else None

        # 给user intent创建 one-hot 向量
        user_act_rep = np.zeros((self.num_intents,))
        user_act_rep[self.intents_dict[user_action['intent']]] = 1.0

        # 给user inform slots 创建向量
        user_inform_slots_rep = np.zeros((self.num_slots,))
        for key in user_action['inform_slots'].keys():
            user_inform_slots_rep[self.slots_dict[key]] = 1.0

        # 给user request slots 创建向量
        user_request_slots_rep = np.zeros((self.num_slots,))
        for key in user_action['request_slots'].keys():
            user_request_slots_rep[self.slots_dict[key]] = 1.0

        # 根据 current_slots 创建已被填充的slots 信息向量
        current_slots_rep = np.zeros((self.num_slots,))
        for key in self.current_informs:
            current_slots_rep[self.slots_dict[key]] = 1.0

        # 给agent intent创建 one-hot 向量
        agent_act_rep = np.zeros((self.num_intents,))
        if last_agent_action:
            agent_act_rep[self.intents_dict[last_agent_action['intent']]] = 1.0

        # 给agent inform slots 创建向量
        agent_inform_slots_rep = np.zeros((self.num_slots,))
        if last_agent_action:
            for key in last_agent_action['inform_slots'].keys():
                agent_inform_slots_rep[self.slots_dict[key]] = 1.0

        # 给agent request slots 创建向量
        agent_request_slots_rep = np.zeros((self.num_slots,))
        if last_agent_action:
            for key in last_agent_action['request_slots'].keys():
                agent_request_slots_rep[self.slots_dict[key]] = 1.0

        # 给当前对话轮次创那one-hot向量
        turn_rep = np.zeros((1,)) + self.round_num / 5.
        turn_onehot_rep = np.zeros((self.max_round_num,))
        turn_onehot_rep[self.round_num - 1] = 1.0

        # 给db的查询结果创建向量 (scaled counts)
        kb_count_rep = np.zeros((self.num_slots + 1,)) + db_results_dict['matching_all_constraints'] / 100.
        for key in db_results_dict.keys():
            if key in self.slots_dict:
                kb_count_rep[self.slots_dict[key]] = db_results_dict[key] / 100.

        # 给db的查询结果创建向量 (binary)
        kb_binary_rep = np.zeros((self.num_slots + 1,)) + np.sum(db_results_dict['matching_all_constraints'] > 0.)
        for key in db_results_dict.keys():
            if key in self.slots_dict:
                kb_binary_rep[self.slots_dict[key]] = np.sum(db_results_dict[key] > 0.)

        # 将以上所有信息拼接
        state_representation = np.hstack(
            [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
             agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep,
             kb_count_rep]).flatten()

        return state_representation

    def update_state_agent(self, agent_action):
        """
        Updates the dialogue history with the agent's action and augments the agent's action.

        Takes an agent action and updates the history. Also augments the agent_action param with query information and
        any other necessary information.
        根据agent action更新对话历史(history)

        参数:
            agent_action (dict): The agent action of format dict('intent': string, 'inform_slots': dict,
                                 'request_slots': dict) and changed to dict('intent': '', 'inform_slots': {},
                                 'request_slots': {}, 'round': int, 'speaker': 'Agent')

        """
        # 当agent intent 为 inform 时，在current_informs的约束条件下，从db中查找inform_slots所有values对应的条目，
        # 取条目最多的value作为inform_slots的值，并将此信息纪录到current_informs中
        if agent_action['intent'] == 'inform':
            assert agent_action['inform_slots']
            inform_slots = self.db_helper.fill_inform_slot(agent_action['inform_slots'], self.current_informs)
            agent_action['inform_slots'] = inform_slots
            assert agent_action['inform_slots']
            key, value = list(agent_action['inform_slots'].items())[0]  # Only one
            assert key != 'match_found'
            assert value != 'PLACEHOLDER', 'KEY: {}'.format(key)
            self.current_informs[key] = value
        # 如果 agent intent 为 match_found， 在current_informs的约束条件下，从db中查找符合条件的条目
        # 随机取一个条目,将此条目的编号作为current_informs中match_key的value
        elif agent_action['intent'] == 'match_found':
            assert not agent_action['inform_slots'], 'Cannot inform and have intent of match found!'
            db_results = self.db_helper.get_db_results(self.current_informs)
            if db_results:
                # Arbitrarily pick the first value of the dict
                key, value = list(db_results.items())[0]
                agent_action['inform_slots'] = copy.deepcopy(value)
                agent_action['inform_slots'][self.match_key] = str(key)
            else:
                agent_action['inform_slots'][self.match_key] = 'no match available'
            self.current_informs[self.match_key] = agent_action['inform_slots'][self.match_key]
        # 更新agent_action中的round_num信息， 并将agent action添加到history
        agent_action.update({'round': self.round_num, 'speaker': 'Agent'})
        self.history.append(agent_action)

    def update_state_user(self, user_action):
        """
        Updates the dialogue history with the user's action and augments the user's action.

        Takes a user action and updates the history. Also augments the user_action param with necessary information.

        参数:
            user_action (dict): The user action of format dict('intent': string, 'inform_slots': dict,
                                 'request_slots': dict) and changed to dict('intent': '', 'inform_slots': {},
                                 'request_slots': {}, 'round': int, 'speaker': 'User')

        """
        # 将 user action中inform_slots的信息添加到current_informs中
        for key, value in user_action['inform_slots'].items():
            self.current_informs[key] = value
        # 更新agent_action中的round_num信息， 并将agent action添加到history
        user_action.update({'round': self.round_num, 'speaker': 'User'})
        self.history.append(user_action)
        # round_num加1
        self.round_num += 1
