# 预处理数据
import pandas as pd

def parse_df(df: pd.DataFrame, span: int):
    # 将DataFrame转换为具有high, low, close, open的格式
    # 这里的df应该是只有一种数据的
    data = {
        'high': [],
        'low': [],
        'open': [],
        'close': []
    }
    for i in range(len(df) // span):
        x = df.iloc[i: i + span]
        data['high'].append(float(x.max()))
        data['low'].append(float(x.min()))
        data['open'].append(float(x.iloc[0]))
        data['close'].append(float(x.iloc[span - 1]))

    return pd.DataFrame(data, index=df.index[:-span:span])