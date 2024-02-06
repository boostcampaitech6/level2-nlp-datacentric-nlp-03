import pandas as pd
from tqdm import tqdm
from transformers import AutoModelWithLMHead, AutoTokenizer

MODEL_NAME = "lcw99/t5-base-korean-paraphrase"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelWithLMHead.from_pretrained(MODEL_NAME)

def t5_paraphrase(text):
    prompt = "paraphrase: " + text
    tokenizer_output = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    # print("encode: >>>", tokenizer_output, tokenizer_output.input_ids)
    generated_ids = model.generate(
        input_ids=tokenizer_output.input_ids,
        max_length=50,  
        num_beams=3,    
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        length_penalty=0.1,
        early_stopping=True
    )
    # print("generated_ids: >>>", generated_ids)
    preds = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
    # print(preds)
    return preds[0]

def t5model_aug(data):
    aug_data = data.copy()
    aug_train = []

    for text in tqdm(aug_data['text']):
        aug_train.append(t5_paraphrase(text))

    aug_data['text'] = aug_train

    new_data = pd.concat([data, aug_data])
    return new_data