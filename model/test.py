# import data_gen
# import pandas as pd


# df = pd.read_csv('data/LBMA-GOLD.csv', parse_dates=True, index_col='Date')
# df.columns = ['Value']
# all_index = pd.date_range('2016-09-12', '2021-09-10', freq='D', name='Date')
# df = df.reindex(all_index)
# df.fillna(inplace=True, method='pad')

# print(df)

# res = data_gen.fill_gold(df)


from gym import spaces
import torch
# a = binary_search(100, 0, 10000, rho=1000, w=[0, 0.5, 0.5], w_=[0.7, 0.2, 0.1], p=[1., 10., 12.], p_=[1., 11., 14.], c=[0., 0.1, 0.1])

print(spaces.Box(0, 1, (2, 2)))
