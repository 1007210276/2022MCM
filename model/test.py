from portfolio_env import get_rho, get_rho_no_gold
import numpy as np

w = np.array([1, 0.0, 0], dtype=np.float32)
w_ = np.array([0.5 ,0.5], dtype=np.float32)
p = np.array([1., 10., 10.], dtype=np.float32)
p_ = np.array([1., 25.], dtype=np.float32)
c = np.array([0., 0.2], dtype=np.float32)

rho_, w_ = get_rho_no_gold(100, 0, 200, 100, w, w_, p, p_, c)
print(100 * w / p)
p_ = np.array([1., 10., 25.], dtype=np.float32)
print(rho_ * w_ / p_)
print(rho_)