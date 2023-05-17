import numpy as np
import pandas as pd
from pytorch_metric_learning import miners, losses
import os
from transformers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pickle
import time
from sklearn.model_selection import train_test_split, KFold
import json
from tqdm.auto import tqdm

data = []

with open("law_candidates_60K.jsonl") as f:
    for line in f:
        data.append(json.loads(line))

queries, passages, answers = [], [], []
for i in tqdm(range(len(data)), position=0, leave=True):
    query = data[i]["summary"]
    passage = data[i]["text"]
    answer = data[i]["answer"]
    queries.append(query)
    passages.append(passage)
    answers.append(answer)

all_data = pd.DataFrame(list(zip(queries, passages, answers)), columns=["queries", "passages", "answers"])

model_name = "monologg/kobigbird-bert-base"
q_tokenizer = AutoTokenizer.from_pretrained(model_name)
p_tokenizer = AutoTokenizer.from_pretrained(model_name)

train_df, test_df = train_test_split(all_data, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

class PairData(Dataset):
    def __init__(self, df: pd.DataFrame):
        super(PairData, self).__init__()
        self.data = df
    def __getitem__(self, index):
        return self.data.iloc[index]
    def __len__(self):
        return self.data.shape[0]

class custom_collate(object):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")
        self.chunk_size = 512
    def __call__(self, batch):
        input_ids, attn_masks, labels = [], [], [] 
        ids = 0
        all_queries = [] 
        for idx, row in enumerate(batch):
            query, passage = row[0], row[1] 
            encoded_q = self.tokenizer(query, max_length=self.chunk_size, padding="max_length", truncation=True, return_tensors="pt") 
            encoded_p = self.tokenizer(passage, max_length=self.chunk_size, padding="max_length", truncation=True, return_tensors="pt") 
            input_ids.append(encoded_q["input_ids"]) 
            attn_masks.append(encoded_q["attention_mask"]) 
            labels.append(ids) 
            
            input_ids.append(encoded_p["input_ids"]) 
            attn_masks.append(encoded_p["attention_mask"]) 
            labels.append(ids) 
            ids += 1 
        input_ids = torch.stack(input_ids, dim=0).squeeze(dim=1)  
        attn_masks = torch.stack(attn_masks, dim=0).squeeze(dim=1)  
        labels = torch.tensor(labels, dtype=int)  
        
        return input_ids, attn_masks, labels  
