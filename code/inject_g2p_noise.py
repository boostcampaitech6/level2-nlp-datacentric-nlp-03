# pip install g2pk
import pandas as pd
from g2pk import G2p

def create_g2p_samples(df:pd.DataFrame, rate:float=0.1):
    num_samples = int(((rate/(1-rate)) * len(df)*2) //2)
    preG2p = df.sample(num_samples, replace=False)
    desG2p = df.sample(num_samples, replace=False)
    
    g2p = G2p()
    
    preG2p['text'] = preG2p['text'].apply(g2p)
    desG2p['text'] = desG2p['text'].apply(lambda x: g2p(x, descriptive=True))
    
    df_g2p = pd.concat([preG2p, desG2p], axis=0)
    
    return df_g2p