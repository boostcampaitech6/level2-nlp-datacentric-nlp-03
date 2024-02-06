from cleanlab.filter import find_label_issues
from cleanlab.dataset import health_summary
import os, random
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# GPU가 사용 가능한지 확인하고, 사용 가능 시 DEVICE 변수에 할당
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {DEVICE}')

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
# OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

# VS_text_entailment, VS_span_inference, VS_span_extraction, VS_unanswerable
# TS_span_inference, TS_span_extraction, TS_unanswerable
CSV_NAME = 'new_g2p_train_without_noise_aihub_expanded_cl_removed.csv'

model_name = 'klue/bert-base'

# 모델 아키텍처를 정의하고 초기화 (훈련할 때와 동일하게).
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)

# `pth` 파일에서 모델 상태 호출
# model.load_state_dict(torch.load('/data/ephemeral/model/original.pth', map_location=DEVICE))
model.load_state_dict(torch.load('/data/ephemeral/model/train_without_noise.pth', map_location=DEVICE))

# tokenizer 호출
tokenizer = AutoTokenizer.from_pretrained(model_name)

class BERTDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 인덱스에 해당하는 데이터 포인트를 가져오기
        text = self.dataframe.iloc[idx]['text']
        label = self.dataframe.iloc[idx]['target']
        
        # 텍스트 토크나이징
        tokenized_input = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = tokenized_input['input_ids'].squeeze(0)
        attention_mask = tokenized_input['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label)
        }

# 모델을 평가 모드로 전환
model.eval()

data = pd.read_csv(os.path.join(DATA_DIR, CSV_NAME))
    
data_bert = BERTDataset(data, tokenizer)

# 예측 확률을 저장할 배열 초기화
train_pred_probs = []

# 배치 사이즈와 데이터 로더를 설정
batch_size = 32  # 예시 배치 사이즈
data_loader = DataLoader(dataset=data_bert, batch_size=batch_size, shuffle=False)

# DataLoader를 통해 배치 단위로 예측
for batch in tqdm(data_loader):
    # 배치에서 입력 데이터 추출
    inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != 'targets'}
    
    # 예측 수행
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 소프트맥스 함수를 적용하여 확률 얻기
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    
    # CPU로 이동 후 NumPy 배열로 변환하여 저장
    train_pred_probs.append(probabilities.cpu().numpy())

# 배열들을 하나의 배열로 병합
train_pred_probs = np.concatenate(train_pred_probs, axis=0)

# find_label_issues를 통해 라벨 문제 찾기
ordered_label_issues = find_label_issues(
    labels=data['target'].to_numpy(),  # numpy array 형태로 전환
    pred_probs=train_pred_probs,
    return_indices_ranked_by='self_confidence'
)

# 예측된 클래스와 확률을 추출합니다.
predicted_classes = np.argmax(train_pred_probs, axis=1)
predicted_probabilities = np.round(np.max(train_pred_probs, axis=1), 2)

for index in ordered_label_issues:
    data.at[index, 'predicted_label'] = predicted_classes[index]
    data.at[index, 'predicted_probability'] = predicted_probabilities[index]


# 라벨 문제가 탐지된 상위 3개를 출력하기
for index in ordered_label_issues[:3]:
    print('input text:', data.iloc[index]['text'])
    print('label:', data.iloc[index]['target'])
    print("------------------")

class_names=[0,1,2,3,4,5,6]

print(CSV_NAME)
health_summary(data['target'], train_pred_probs, class_names=class_names)

# DataFrame 방식이 아니라 생성된 리스트를 바로 사용하여 인덱스 추출
indices_to_remove = ordered_label_issues

# 문제가 있는 항목들만 필터링하여 새로운 DataFrame 생성
train_df_issues = data.loc[indices_to_remove]

# 문제가 있는 항목들을 CSV로 저장
# train_df_issues.to_csv(os.path.join(DATA_DIR, 'issues_'+CSV_NAME), index=False)
train_df_issues.to_csv(os.path.join(DATA_DIR, 'wissues_'+CSV_NAME), index=False)

# 해당 인덱스의 행들을 df에서 삭제
train_df_cl_removed = data.drop(labels=indices_to_remove).reset_index(drop=True)

# train_df_cl_removed.to_csv(os.path.join(DATA_DIR, 'cl_'+CSV_NAME), index=False) 
train_df_cl_removed.to_csv(os.path.join(DATA_DIR, 'wcl_'+CSV_NAME), index=False) 
