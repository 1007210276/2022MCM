import pandas as pd
import numpy as np


def evaluate(model, env, states=None):
    obs = env.reset()
    rewards = []
    while True:
        action, _ = model.predict(obs)
        obs, reward, dones, info = env.step(action)
        if dones:
            break
        rewards.append(reward)
        if states is not None:
            states.append(info[0]['state'])

    r = np.array(rewards)
    r = r.cumsum()
    return np.exp(r)
