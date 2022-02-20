import pandas as pd
bitcoin = pd.read_csv('../data/BCHAIN-MKPRU.csv', index_col='Date', parse_dates=True)
gold = pd.read_csv('../data/LBMA-GOLD.csv', index_col='Date', parse_dates=True)
gold.columns = ['Value']
all_index = pd.date_range('2016-9-12', '2021-9-10', freq='D')
gold = gold.reindex(all_index)
price = pd.concat([gold.fillna(method='pad'), bitcoin[1:]], axis=1)
price.columns = ['gold', 'bitcoin']