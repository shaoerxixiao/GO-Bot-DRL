from collections import defaultdict
from dialogue_config import no_query_keys, usersim_default_key
import copy


class DBQuery:
    """查询数据库，为状态追踪器（state tracker）提供信息"""

    def __init__(self, database):
        """
        参数：以dict方式存储的关于电影信息的database
        """

        self.database = database
        # {frozenset: {string: int}} A dict of dicts
        self.cached_db_slot = defaultdict(dict)
        # {frozenset: {'#': {'slot': 'value'}}} A dict of dicts of dicts, a dict of DB sub-dicts
        self.cached_db = defaultdict(dict)
        # 不需要查询的keys
        self.no_query = no_query_keys
        self.match_key = usersim_default_key

    def fill_inform_slot(self, inform_slot_to_fill, current_inform_slots):
        """
        Given the current informs/constraints fill the informs that need to be filled with values from the database.

        Searches through the database to fill the inform slots with PLACEHOLDER with values that work given the current
        constraints of the current episode.

        参数:
            inform_slot_to_fill (dict): 需要查询values的Inform slots
            current_inform_slots (dict): StateTracker中现有的已知values的inform slots

        返回:
            dict: inform_slot_to_fill 被填充好values的inform_slot_to_fill
        """

        # 在这个框架里，每一回合里inform slot 只允许有一个
        assert len(inform_slot_to_fill) == 1
        # 取第一个（也是唯一的一个）key,即词槽
        key = list(inform_slot_to_fill.keys())[0]

        # 深拷贝current_inform_slots为current_informs
        # 如里key在current_inform_slots中已存在，则在current_inform_slots中去除这个key
        # 这样这个key就可以被重复查询
        current_informs = copy.deepcopy(current_inform_slots)
        current_informs.pop(key, None)

        # 在current_informs的条件下，返回符合条件的信息，相当于返回结果是一个db的subset
        db_results = self.get_db_results(current_informs)

        # 利用_count_slot_values函数在db中查询key可能的取值以及相应的数量，返回以values-occurrences为键值对的dict
        # 将occurrence最大的value 作为key 的 value，如果没有则no match available
        filled_inform = {}
        values_dict = self._count_slot_values(key, db_results)
        if values_dict:
            # 取occurrence最大的value作为key的value
            filled_inform[key] = max(values_dict, key=values_dict.get)
        else:
            filled_inform[key] = 'no match available'

        return filled_inform

    def _count_slot_values(self, key, db_subdict):
        """
        key为词槽，查询此词槽在db_subdict中对应的取值以及不同取值对应的出现次数
        参数:
            key (string): 需要查询计数信息的key
            db_subdict (dict): database 的一部分

        返回:
            dict: key 所对应的 values 以及 occurrences
        """

        slot_values = defaultdict(int)  # init to 0
        for id in db_subdict.keys():
            current_option_dict = db_subdict[id]
            # If there is a match
            if key in current_option_dict.keys():
                slot_value = current_option_dict[key]
                # This will add 1 to 0 if this is the first time this value has been encountered, or it will add 1
                # to whatever was already in there
                slot_values[slot_value] += 1
        return slot_values

    def get_db_results(self, constraints):
        """
        在现有的约束条件下，查询database，返回所有满足条件的电影信息

        参数:
            constraints (dict): 现有的informs信息

        返回:
            dict: 满足条件的电影信息
        """

        # 过滤掉不需要查询的keys以及value为'anything'的keys
        new_constraints = {k: v for k, v in constraints.items() if k not in self.no_query and v is not 'anything'}

        inform_items = frozenset(new_constraints.items())
        cache_return = self.cached_db[inform_items]

        if cache_return == None:
            # If it is none then no matches fit with the constraints so return an empty dict
            return {}
        # if it isnt empty then return what it is
        if cache_return:
            return cache_return
        # else continue on

        available_options = {}
        for id in self.database.keys():
            current_option_dict = self.database[id]
            # First check if that database item actually contains the inform keys
            # Note: this assumes that if a constraint is not found in the db item then that item is not a match
            if len(set(new_constraints.keys()) - set(self.database[id].keys())) == 0:
                match = True
                # Now check all the constraint values against the db values and if there is a mismatch don't store
                for k, v in new_constraints.items():
                    if str(v).lower() != str(current_option_dict[k]).lower():
                        match = False
                if match:
                    # Update cache
                    self.cached_db[inform_items].update({id: current_option_dict})
                    available_options.update({id: current_option_dict})

        # if nothing available then set the set of constraint items to none in cache
        if not available_options:
            self.cached_db[inform_items] = None

        return available_options

    def get_db_results_for_slots(self, current_informs):
        """
        给定现有的 inform slot (key and value)，计算满足条件的database中的item数目

        参数:
            current_informs (dict): 现有的约束条件，形式为slot-value对
        返回:
            dict: current_informs中的每个约束条件所对应的符合条件的item数目
        """

        # The items (key, value) of the current informs are used as a key to the cached_db_slot
        inform_items = frozenset(current_informs.items())
        # A dict of the inform keys and their counts as stored (or not stored) in the cached_db_slot
        cache_return = self.cached_db_slot[inform_items]

        if cache_return:
            return cache_return

        # If it made it down here then a new query was made and it must add it to cached_db_slot and return it
        # Init all key values with 0
        db_results = {key: 0 for key in current_informs.keys()}
        db_results['matching_all_constraints'] = 0

        for id in self.database.keys():
            all_slots_match = True
            for CI_key, CI_value in current_informs.items():
                # Skip if a no query item and all_slots_match stays true
                if CI_key in self.no_query:
                    continue
                # If anything all_slots_match stays true AND the specific key slot gets a +1
                if CI_value == 'anything':
                    db_results[CI_key] += 1
                    continue
                if CI_key in self.database[id].keys():
                    if CI_value.lower() == self.database[id][CI_key].lower():
                        db_results[CI_key] += 1
                    else:
                        all_slots_match = False
                else:
                    all_slots_match = False
            if all_slots_match: db_results['matching_all_constraints'] += 1

        # update cache (set the empty dict)
        self.cached_db_slot[inform_items].update(db_results)
        assert self.cached_db_slot[inform_items] == db_results
        return db_results
