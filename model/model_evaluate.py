import pandas as pd
import numpy as np


def evaluate(model, env):
    obs = env.reset()
    rewards = []
    while True:
        action, _ = model.predict(obs)
        obs, reward, dones, _ = env.step(action)
        if dones:
            break
        rewards.append(reward)

    r = np.array(rewards)
    r = r.cumsum()
    return np.exp(r)
