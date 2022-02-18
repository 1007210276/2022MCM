import gym
from gym import spaces
import numpy as np
import pandas as pd

def get_rho(batch: int, f, t,  rho, w, w_, p, p_, c):
    rho = np.array(rho)
    w = np.array(w)
    w_ = np.array(w_)
    p = np.array(p)
    p_ = np.array(p_)
    c = np.array(c)
    def rho_func(rho_):
        delta = rho_ * w_ / p_ - rho * w  / p
        p_[delta < 0] *= (1 - c)[delta < 0]
        p_[delta > 0] /= (1 - c)[delta > 0]
        return p_.dot(delta)

    mid = (f + t) / 2
    for _ in range(batch):
        mid = (f + t) / 2
        if rho_func(mid) > 0:
            t = mid
        else:
            f = mid
    return mid

class PortfolioEnv(gym.Env):
    def __init__(self, 
    price_gold: pd.DataFrame, 
    price_bitcoin: pd.DataFrame, 
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cur_date: pd.Timestamp = None) -> None:
        super().__init__()
        self.price_gold = price_gold
        self.price_bitcoin = price_bitcoin
        self.start_date = start_date
        self.end_date = end_date
        if not cur_date:
            cur_data = self.start_date

    @property
    def action_space(self):
        # Calculate whether we can trade gold on self.cur_date 
        # and returns appropriate action space 
        pass
    
    def reset(self, day: pd.Timestamp = None):
        self.day = day if day else self.start_date
        # return observation, reward, done, info
        return 

    def step(self, action):
        pass