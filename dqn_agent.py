from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random, copy
import numpy as np
from dialogue_config import rule_requests, agent_actions
import re


class DQNAgent:
    """强化学习模型"""

    def __init__(self, state_size, constants):
        """
        The constructor of DQNAgent.

        The constructor of DQNAgent which saves constants, sets up neural network graphs, etc.

        参数:
            state_size (int): 状态维度
            constants (dict): 配置参数

        """

        self.C = constants['agent']
        self.memory = []
        self.memory_index = 0
        self.max_memory_size = self.C['max_mem_size']
        self.eps = self.C['epsilon_init']
        self.vanilla = self.C['vanilla']
        self.lr = self.C['learning_rate']
        self.gamma = self.C['gamma']
        self.batch_size = self.C['batch_size']
        self.hidden_size = self.C['dqn_hidden_size']

        self.load_weights_file_path = self.C['load_weights_file_path']
        self.save_weights_file_path = self.C['save_weights_file_path']

        if self.max_memory_size < self.batch_size:
            raise ValueError('Max memory size must be at least as great as batch size!')

        self.state_size = state_size
        self.possible_actions = agent_actions
        self.num_actions = len(self.possible_actions)

        self.rule_request_set = rule_requests

        self.beh_model = self._build_model()
        self.tar_model = self._build_model()

        self._load_weights()

        self.reset()

    def _build_model(self):
        """创建NN模型，输入为state representation，输出为action"""

        model = Sequential()
        model.add(Dense(self.hidden_size, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def reset(self):
        """Resets the rule-based variables."""

        self.rule_current_slot_index = 0
        self.rule_phase = 'not done'

    def get_action(self, state, use_rule=False):
        """
        根据state返回agent action
        两种可选策略：随机生成action;
                    rule-based policy（基于规则）或者 neural networks（基于深度学习网络）来选择 action

        参数:
            state (numpy.array): The database with format dict(long: dict)
            use_rule (bool): 指明是否使用 rule-based policy, 默认为False。
                             取决于使用warmup模式（取True）还是training模式(取False).

        返回:
            int: action的标号
            dict: action/response

        """

        if self.eps > random.random():
            index = random.randint(0, self.num_actions - 1)
            action = self._map_index_to_action(index)
            return index, action
        else:
            if use_rule:
                return self._rule_action()
            else:
                return self._dqn_action(state)

    def _rule_action(self):
        """
        返回基于rule-based policy 得到的action

        Returns:
            int: action的标号
            dict: action/response

        """

        if self.rule_current_slot_index < len(self.rule_request_set):
            slot = self.rule_request_set[self.rule_current_slot_index]
            self.rule_current_slot_index += 1
            rule_response = {'intent': 'request', 'inform_slots': {}, 'request_slots': {slot: 'UNK'}}
        elif self.rule_phase == 'not done':
            rule_response = {'intent': 'match_found', 'inform_slots': {}, 'request_slots': {}}
            self.rule_phase = 'done'
        elif self.rule_phase == 'done':
            rule_response = {'intent': 'done', 'inform_slots': {}, 'request_slots': {}}
        else:
            raise Exception('Should not have reached this clause')

        index = self._map_action_to_index(rule_response)
        return index, rule_response

    def _map_action_to_index(self, response):
        """
        输出action对应的序号

        参数:
            response (dict)，即为一个action

        返回:
            int
        """

        for (i, action) in enumerate(self.possible_actions):
            if response == action:
                return i
        raise ValueError('Response: {} not found in possible actions'.format(response))

    def _dqn_action(self, state):
        """
        返回neural networks（基于深度学习网络）预测得到的 action

        参数:
            state (numpy.array)

        返回:
            int: action的标号
            dict: action/response
        """

        index = np.argmax(self._dqn_predict_one(state))
        action = self._map_index_to_action(index)
        return index, action

    def _dqn_predict_one(self, state, target=False):
        """
        利用neural networks，根据state预测action （一个输入）

        参数:
            state (numpy.array)
            target (bool)

        返回:
            numpy.array
        """

        return self._dqn_predict(state.reshape(1, self.state_size), target=target).flatten()

    def _dqn_predict(self, states, target=False):
        """
        利用neural networks，根据state预测action （多个输入）

        参数:
            state (numpy.array)
            target (bool)

        返回:
            numpy.array
        """

        if target:
            return self.tar_model.predict(states)
        else:
            return self.beh_model.predict(states)

    def _map_index_to_action(self, index):
        """
        输出序号对应的action

        参数:
            index (int)

        返回:
            dict
        """

        for (i, action) in enumerate(self.possible_actions):
            if index == i:
                return copy.deepcopy(action)
        raise ValueError('Index: {} not in range of possible actions'.format(index))


    def add_experience(self, state, action, reward, next_state, done):
        """
        将experience（包括 state, action, reward, next_state, done） 添加至 memory
        存储方式为memory[memory_index] = (state, action, reward, next_state, done)
        参数:
            state (numpy.array) 当前状态
            action (int) 行为
            reward (int) 反馈
            next_state (numpy.array) 下一个状态
            done (bool) 是否完成对话

        """

        if len(self.memory) < self.max_memory_size:
            self.memory.append(None)
        self.memory[self.memory_index] = (state, action, reward, next_state, done)
        self.memory_index = (self.memory_index + 1) % self.max_memory_size

    def empty_memory(self):
        """清空 memory """

        self.memory = []
        self.memory_index = 0

    def is_memory_full(self):
        """查看memory是否已满"""

        return len(self.memory) == self.max_memory_size

    def train(self):
        """
        Trains the agent by improving the behavior model given the memory tuples.

        Takes batches of memories from the memory pool and processing them. The processing takes the tuples and stacks
        them in the correct format for the neural network and calculates the Bellman equation for Q-Learning.

        """

        # 计算batch数量，num_batches = len(memory) // batch_size
        num_batches = len(self.memory) // self.batch_size
        for b in range(num_batches):
            # 从memory里随机取batch_size大小的样例
            batch = random.sample(self.memory, self.batch_size)
            # 取出样例中的states以及next_states
            states = np.array([sample[0] for sample in batch])
            next_states = np.array([sample[3] for sample in batch])

            assert states.shape == (self.batch_size, self.state_size), 'States Shape: {}'.format(states.shape)
            assert next_states.shape == states.shape
            # 根据states,利用深度模型预测action
            beh_state_preds = self._dqn_predict(states)  # For leveling error
            # vanilla表示用DQN, not vanilla表示用 Double DQN
            if not self.vanilla:
                beh_next_states_preds = self._dqn_predict(next_states)  # For indexing for DDQN
            tar_next_state_preds = self._dqn_predict(next_states, target=True)  # For target value for DQN (& DDQN)

            inputs = np.zeros((self.batch_size, self.state_size))
            targets = np.zeros((self.batch_size, self.num_actions))

            for i, (s, a, r, s_, d) in enumerate(batch):
                t = beh_state_preds[i]
                if not self.vanilla:
                    t[a] = r + self.gamma * tar_next_state_preds[i][np.argmax(beh_next_states_preds[i])] * (not d)
                else:
                    t[a] = r + self.gamma * np.amax(tar_next_state_preds[i]) * (not d)

                inputs[i] = s
                targets[i] = t

            self.beh_model.fit(inputs, targets, epochs=1, verbose=0)

    def copy(self):
        """将behavior model的参数权重复制到target model中"""

        self.tar_model.set_weights(self.beh_model.get_weights())

    def save_weights(self):
        """保存模型参数权重"""

        if not self.save_weights_file_path:
            return
        beh_save_file_path = re.sub(r'\.h5', r'_beh.h5', self.save_weights_file_path)
        self.beh_model.save_weights(beh_save_file_path)
        tar_save_file_path = re.sub(r'\.h5', r'_tar.h5', self.save_weights_file_path)
        self.tar_model.save_weights(tar_save_file_path)

    def _load_weights(self):
        """ 加载模型参数权重 """

        if not self.load_weights_file_path:
            return
        beh_load_file_path = re.sub(r'\.h5', r'_beh.h5', self.load_weights_file_path)
        self.beh_model.load_weights(beh_load_file_path)
        tar_load_file_path = re.sub(r'\.h5', r'_tar.h5', self.load_weights_file_path)
        self.tar_model.load_weights(tar_load_file_path)
