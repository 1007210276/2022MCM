import pandas as pd
import numpy as np

class HumanAgent():
    def __init__(self, date: pd.Timestamp):
        self.date = date
        self.count = 0

    def predict(self, obs):
        self.count += 1
        if self.count == 1:
            return np.array([20, -20., -20., 20.], dtype=np.float32), {}
        elif self.count == 480:
            return np.array([20, -20., 20., -20.], dtype=np.float32), {}
        elif self.count == 760:
            return np.array([20, -20., -20., 20.], dtype=np.float32), {}
        elif self.count == 1600:
            return np.array([20, -20., 20., -20.], dtype=np.float32), {}
        elif self.count == 1200:
            return np.array([20, -20., 20., -20.], dtype=np.float32), {}
        elif self.count == 1350:
            return np.array([20, -20., -20., 20.], dtype=np.float32), {}
        else:
            return np.array([-2, -20., -20., 20.], dtype=np.float32), {}
        