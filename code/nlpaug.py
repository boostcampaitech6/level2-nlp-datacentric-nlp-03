# pip install nlpaug
import pandas as pd
import nlpaug.augmenter.word as naw

def get_word_embs_aug(df, aug_p=0.1):
    aug = naw.ContextualWordEmbsAug(model_path='klue/bert-base', aug_p=aug_p)
    df_aug = df.copy()
    df_aug['text'] = df_aug['text'].apply(lambda x: aug.augment(x)[0])
    return df_aug