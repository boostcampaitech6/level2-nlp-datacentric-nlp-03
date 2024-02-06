import googletrans # pip install googletrans==4.0.0-rc1
import pandas as pd
import os

def BTS(input_str, translator):
    # target language로의 번역 수행
    temp_result = translator.translate(input_str, dest='en').text
    #temp_result = translator.translate(input_str, dest='ja').text
    # 다시 source language로의 번역 수행
    result = translator.translate(temp_result, dest='ko').text
    return result

# 사용할 번역기 api 로드
translator = googletrans.Translator()

# 증강할 데이터 로드
BASE_DIR = os.getcwd()
dataset = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))

ids = dataset['ID']
text = dataset['text']
targets = dataset['target']
urls = dataset['url']
dates = dataset['date']

df = pd.DataFrame({'id':ids, 'text':text,
'target': targets, 'url': urls, 'date':dates})

augmented_data, augmented_ind, wrong_ind = [], [], []
for i in range(len(df)):
    try:
        data = df['input_text'].iloc[i]
        augmented = BTS(data, translator)
        #print(i, augmented)
        augmented_data.append(augmented)
        augmented_ind.append(i)
    except:
        wrong_ind.append(i)
        #print(i, "wrong")

df = df.iloc[augmented_ind]
df['input_text'] = augmented_data
df.to_csv(os.path.join(BASE_DIR,'train_BT.csv'), index = False)

# 원본 데이터와 합치기
BASE_DIR = os.getcwd()

df1 = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
df2 = pd.read_csv(os.path.join(BASE_DIR, 'train_BT.csv'))

result = pd.concat([df1,df2])

result.drop_duplicates(subset = 'text') # 중복 제거
result.info()

result.to_csv(os.path.join(BASE_DIR, 'train_BT_integrated.csv'), index = False)

