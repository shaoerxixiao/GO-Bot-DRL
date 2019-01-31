# 一些特殊的词槽值
'PLACEHOLDER'  # inform slots里会出现的value，表示待查询
'UNK'  # request slots里会出现的value， 表示目前未知，待询问
'anything'  # value为anything，表示对取值无所谓
'no match available'  # 此时agent的intent 为 match_found，但是 db 里找不到满足约束条件的信息

#######################################
# Usersim Config 用户模拟的相关配置
#######################################
# Used in EMC for intent error (and in user)
usersim_intents = ['inform', 'request', 'thanks', 'reject', 'done']

# agent 的目标是为这个 key 找到相应的值
usersim_default_key = 'ticket'

# Required to be in the first action in inform slots of the usersim if they exist in the goal inform slots
usersim_required_init_inform_keys = ['moviename']

#######################################
# Agent Config 机器的相关配置
#######################################

# 所有可能的inform 以及 request slots
agent_inform_slots = ['moviename', 'theater', 'starttime', 'date', 'genre', 'state', 'city', 'zip', 'critic_rating',
                      'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price', 'actor',
                      'description', 'other', 'numberofkids', usersim_default_key]
agent_request_slots = ['moviename', 'theater', 'starttime', 'date', 'numberofpeople', 'genre', 'state', 'city', 'zip',
                       'critic_rating', 'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price',
                       'actor', 'description', 'other', 'numberofkids']

# 所有可能的 actions
agent_actions = [
    {'intent': 'done', 'inform_slots': {}, 'request_slots': {}},  # Triggers closing of conversation
    {'intent': 'match_found', 'inform_slots': {}, 'request_slots': {}}
]
for slot in agent_inform_slots:
    # Must use intent match found to inform this, but still have to keep in agent inform slots
    if slot == usersim_default_key:
        continue
    agent_actions.append({'intent': 'inform', 'inform_slots': {slot: 'PLACEHOLDER'}, 'request_slots': {}})
for slot in agent_request_slots:
    agent_actions.append({'intent': 'request', 'inform_slots': {}, 'request_slots': {slot: 'UNK'}})

# 基于规则的回答策略中的request list
rule_requests = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']

# These are possible inform slot keys that cannot be used to query
no_query_keys = ['numberofpeople', usersim_default_key]

#######################################
# Global config 全局配置
#######################################

# These are used for both constraint check AND success check in usersim
FAIL = -1
NO_OUTCOME = 0
SUCCESS = 1

# 所有的 intents (for one-hot conversion in ST.get_state())
all_intents = ['inform', 'request', 'done', 'match_found', 'thanks', 'reject']

# 所有的 slots (for one-hot conversion in ST.get_state())
all_slots = ['actor', 'actress', 'city', 'critic_rating', 'date', 'description', 'distanceconstraints',
             'genre', 'greeting', 'implicit_value', 'movie_series', 'moviename', 'mpaa_rating',
             'numberofpeople', 'numberofkids', 'other', 'price', 'seating', 'starttime', 'state',
             'theater', 'theater_chain', 'video_format', 'zip', 'result', usersim_default_key, 'mc_list']
