from gym.envs.registration import register

register(
    id='PandaKitchen-v0',
    entry_point='panda_robot_sim.envs:PandaKitchenEnv'
)
register(
    id='PandaKitchen-v1',
    entry_point='panda_robot_sim.envs:PandaKitchenEnv_v1'
)
register(
    id='PandaKitchen-v2',
    entry_point='panda_robot_sim.envs:PandaKitchenEnv_v2'
)
register(
    id='PandaKitchen-v3',
    entry_point='panda_robot_sim.envs:PandaKitchenEnv_goal'
)
register(
    id='PandaKitchen-v4',
    entry_point='panda_robot_sim.envs:PandaKitchenEnv_goal_2'
)
register(
    id='PandaKitchen-v5',
    entry_point='panda_robot_sim.envs:PandaKitchenEnv_no_goal'
)