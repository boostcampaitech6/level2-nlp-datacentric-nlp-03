from data_loader import AugmentDataSet
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForMaskedLM
import tqdm
import torch
from torch.utils.data import DataLoader
import copy
from typing import Union


def mask_tokens(tokenizer, input_ids:torch.Tensor, mlm_prob:float=0.15, do_rep_random:bool=True):
    '''
        Copied from huggingface/transformers/data/data_collator - torch.mask_tokens()
        Prepare masked tokens inputs/labels for masked language modeling
        if do_rep_random is True:
            80% MASK, 10% random, 10% original
        else:
            100% MASK
    '''
    labels = input_ids.clone()

    probability_matrix = torch.full(labels.shape, mlm_prob)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value = 0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100 # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    mask_rep_prob = 0.8
    if not do_rep_random:
        mask_rep_prob = 1.0
    
    indices_replaced = torch.bernoulli(torch.full(labels.shape, mask_rep_prob)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    if do_rep_random:
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

    return input_ids, labels

def candidate_filtering(tokenizer:AutoTokenizer,
                        input_ids:list,
                        idx:int,
                        org:int,
                        candidates:Union[list, torch.Tensor]) -> int:
    '''
    후보 필터링 조건에 만족하는 최적의 후보 선택
    1. 원래 토큰과 후보 토큰이 같은 타입(is_same_token_type 참고)
    2. 현 위치 앞 혹은 뒤에 동일한 토큰이 있지 않음
    '''

    org_token = tokenizer.convert_ids_to_tokens([org])[0]
    candidate_tokens = tokenizer.convert_ids_to_tokens(candidates.cpu().tolist())

    for rank, token in enumerate(candidate_tokens):
        if org_token!=token and is_same_token_type(org_token, token):
            if input_ids[idx-1]==candidates[rank] or input_ids[idx+1]==candidate_tokens[rank]:
                continue
            return candidates[rank]

    return org

def is_same_token_type(org_token:str, candidate:str) -> bool:
    '''
    후보 필터링 조건을 만족하는지 확인
    - 후보와 원 토큰의 타입을 문장부호와 일반 토큰으로 나누어 같은 타입에 속하는지 확인
    '''
    res = False
    if org_token[0]=="#" and org_token[2:].isalpha()==candidate.isalpha():
        res = True
    elif candidate[0]=="#" and org_token.isalpha()==candidate[2:].isalpha():
        res = True
    elif candidate[0]=="#" and org_token[0]=="#" and org_token[2:].isalpha()==candidate[2:].isalpha():
        res = True
    elif org_token.isalpha()==candidate.isalpha() and (candidate[0]!="#" and org_token[0]!="#"):
        res = True

    return res

def batch_augment(model:AutoModelForMaskedLM,
                tokenizer:AutoTokenizer,
                dataset:torch.utils.data.Dataset,
                k, threshold, mlm_prob, batch_size) -> str:
    '''
    배치 단위의 문장에 랜덤으로 마스킹을 적용하여 새로운 문장 배치를 생성(증강)

    args:
        model(AutoModelForMaskedLM)
        tokenizer(AutoTokenizer)
        dataset(torch.utils.data.Dataset)
        dev(str or torch.device)
        args(argparse.Namespace)
            - k(int, default=5)
            - threshold(float, default=0.95)
           -  mlm_prob(float, default=0.15)
        
    return:
        (list) : 증강한 문장들의 리스트
    '''
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    augmented_res = []
    dataloader = DataLoader(dataset, batch_size = batch_size)
    for batch in tqdm.tqdm(dataloader):
        #########################################################
        # 인풋 문장에 랜덤으로 마스킹 적용
        input_ids, attention_masks = batch[0], batch[1]
        masked_input_ids, _ = mask_tokens(tokenizer, input_ids, mlm_prob, do_rep_random=False)

        masked_input_ids = masked_input_ids.to(dev)
        attention_masks = attention_masks.to(dev)
        labels = input_ids
        #########################################################

        with torch.no_grad():
            output = model(masked_input_ids, attention_mask = attention_masks)
            logits1 = output["logits"]

        #########################################################
        # 배치 내의 문장 별로 후보 필터링을 적용하고, 결과를 토대로 새로운 문장 생성
        augmented1 = []
        for sent_no in range(len(masked_input_ids)):
            copied = copy.deepcopy(input_ids.cpu().tolist()[sent_no])

            for i in range(len(masked_input_ids[sent_no])):
                if masked_input_ids[sent_no][i] == tokenizer.pad_token_id:
                    break

                if masked_input_ids[sent_no][i] == tokenizer.mask_token_id:
                    org_token = labels.cpu().tolist()[sent_no][i]
                    prob = logits1[sent_no][i].softmax(dim=0)
                    probability, candidates = prob.topk(k)
                    if probability[0]<threshold:
                        res = candidate_filtering(tokenizer, copied, i, org_token, candidates)
                    else:
                        res = candidates[0]
                    copied[i] = res

            copied = tokenizer.decode(copied, skip_special_tokens=True)
            augmented1.append(copied)
        #########################################################
        augmented_res.extend(augmented1)

    return augmented_res


def augmentation(file_path:str, output_path:str) -> None:

    input_df = pd.read_csv(file_path)
    text_list = input_df["text"].tolist()

    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-generator")

    dataset = AugmentDataSet(text_list, tokenizer)
    model = AutoModelForMaskedLM.from_pretrained("monologg/koelectra-base-v3-generator")
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    augmented = batch_augment(model, tokenizer, dataset, 5, 0.95, 0.15, 1)

    aug_id_prefix = "aug_"
    aug_url = "mlm_augment"
    aug_date = "20240130"

    augmented_df = pd.DataFrame({"id": [aug_id_prefix+str(i) for i in range(len(augmented))],
                                "text": augmented,
                                "target": input_df["target"].tolist(),
                                "url": [aug_url for i in range(len(augmented))],
                                "date": [aug_date for i in range(len(augmented))]})
    
    concat_df = pd.concat([input_df, augmented_df], axis=0)
    concat_df.to_csv(output_path, index=False)