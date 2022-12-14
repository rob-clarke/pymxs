from gym.envs.registration import register

register(
    id="gym_mxs/MXS-v0",
    entry_point="gym_mxs.envs:MxsEnv",
)
