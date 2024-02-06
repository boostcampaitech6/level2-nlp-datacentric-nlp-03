# pip install g2pk
import pandas as pd
from g2pk import G2p

def split_g2p_noise(df:pd.DataFrame):
    g2p = G2p()

    preG2p = df['text'].apply(g2p)
    desG2p = df['text'].apply(lambda x : g2p(x, descriptive=True))

    isG2pList = []
    def add_noise_column(data):
    if data['text'] == desG2p[data.name]:
        isG2pList.append('descriptive')
    elif data['text'] == preG2p[data.name]:
        isG2pList.append('prescriptive')
    else:
        isG2pList.append('original')

    df.apply(add_noise_column, axis=1)
    train = pd.concat([df, pd.DataFrame({'isG2p' : isG2pList})], axis=1)
    
    train_without_noise = train[train['isG2p']=='original']
    train_only_noise = train[train['isG2p']!='original']
    
    train_without_noise = train_without_noise.drop(columns=['isG2p'])
    train_only_noise = train_only_noise.drop(columns=['isG2p'])

    return train_without_noise, train_only_noise