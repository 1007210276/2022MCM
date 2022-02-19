import gym
from gym import spaces
import numpy as np
import pandas as pd
import copy
import torch


def bsearch(func, batch, f, t, *args, **kwargs):
    mid = (f + t) / 2
    for _ in range(batch):
        mid = (f + t) / 2
        if func(mid, *args, **kwargs) > 0:
            t = mid
        else:
            f = mid
    return mid


def get_rho_no_gold(batch: int, f, t, rho, w, w_, p, p_, c):
    # p_ (2, )
    # w_ (2, )
    # p (3, )
    # w (3, )
    rho = np.array(rho)
    w = np.array(w)
    w_ = np.array(w_)
    p = np.array(p)
    p_ = np.array(p_)
    c = np.array(c)

    gold_value = w[1] * rho

    rho_no_gold = rho - gold_value

    wa = np.array([w[0], w[2]])
    pa = np.array([p[0], p[2]])

    def rho_func(rho_no_gold_):
        delta = rho_no_gold_ * w_ / p_ - rho_no_gold * wa / pa
        copy_p_ = copy.deepcopy(p_)
        copy_p_[delta < 0] *= (1 - c)[delta < 0]
        copy_p_[delta > 0] /= (1 - c)[delta > 0]
        return copy_p_.dot(delta)
    rho_no_gold_ = bsearch(rho_func, batch, f, t)
    rho_aux = rho_no_gold_ * w_
    rho_ = rho_no_gold_ + gold_value
    w_ = np.array([rho_aux[0], gold_value, rho_aux[1]]) / rho_
    return rho_, w_


def get_rho(batch: int, f, t,  rho, w, w_, p, p_, c):
    rho = np.array(rho)
    w = np.array(w)
    w_ = np.array(w_)
    p = np.array(p)
    p_ = np.array(p_)
    c = np.array(c)

    def rho_func(rho_):
        delta = rho_ * w_ / p_ - rho * w / p
        copy_p_ = copy.deepcopy(p_)
        copy_p_[delta < 0] *= (1 - c)[delta < 0]
        copy_p_[delta > 0] /= (1 - c)[delta > 0]
        return copy_p_.dot(delta)

    return bsearch(rho_func, batch, f, t)


class PortfolioEnv(gym.Env):
    def __init__(self,
                 price: pd.DataFrame,
                 gold_trade: pd.Series,
                 start_date: pd.Timestamp,
                 end_date: pd.Timestamp,
                 cur_date: pd.Timestamp = None,
                 observation_length: int = 10,
                 currency: np.float32 = 1000) -> None:
        super().__init__()
        self.price = price
        self.gold_trade = gold_trade
        self.start_date = start_date
        self.end_date = end_date
        self.observation_length = observation_length
        if not cur_date:
            self.cur_date = self.start_date + \
                pd.DateOffset(n=observation_length + 1)
        self.state = np.array([1., 0., 0.])
        self.currency = currency

    @property
    def observation_space(self):
        return spaces.Box(low=0.0, high=1e10, shape=(self.observation_length, 2))

    def _check_gold_trade(self, date: pd.Timestamp) -> bool:
        # Check if it is possible to trade gold on 'date'
        return self.gold_trade.loc[date]

    @property
    def action_space(self):
        # Calculate whether we can trade gold on self.cur_date
        # and returns appropriate action space
        return spaces.Box(low=-50, high=10, shape=[3, ], dtype=np.float32)

    def build_p(self, date: pd.Timestamp):
        pg = self.price['gold'].loc[date]
        return np.array([1, pg, self.price['bitcoin'].loc[date]])

    def build_observation(self):
        r = self.price.loc[self.cur_date -
                           pd.DateOffset(n=self.observation_length - 1): self.cur_date]
        return r.to_numpy()

    def step(self, action):
        kwargs = {
            'w': self.state,
            'p': self.build_p(self.cur_date),
            'p_': self.build_p(self.cur_date + pd.DateOffset(n=1)),
            'c': [0, 0.01, 0.02],
            'rho': self.currency
        }
        if self._check_gold_trade(self.cur_date):
            action = np.exp(action) / sum(np.exp(action))
            kwargs['w_'] = action
            rho_ = get_rho(100, 0, self.currency * 5, **kwargs)
            reward = np.log(rho_ / self.currency)
            self.state = action
        else:
            action = np.array([action[0], action[2]], dtype=np.float32)
            action = np.exp(action) / sum(np.exp(action))
            kwargs['w_'] = action
            kwargs['c'] = [0, 0.02]
            kwargs['p_'] = np.array(
                [kwargs['p'][0], kwargs['p'][2]], dtype=np.float32)
            rho_, w_ = get_rho_no_gold(100, 0, self.currency * 5, **kwargs)
            reward = np.log(rho_ / self.currency)
            self.state = w_
        self.currency = rho_
        done = self.cur_date >= self.end_date - \
            pd.DateOffset(n=self.observation_length + 20)
        observation = self.build_observation()
        self.cur_date = self.cur_date + pd.DateOffset(n=1)
        return observation, reward, done, {}

    def reset(self):
        self.state = np.array([1., 0., 0.], dtype=np.float32)
        # self.currency = 1000.
        self.cur_date = self.start_date + \
            pd.DateOffset(n=self.observation_length + 1)
        return self.build_observation()
