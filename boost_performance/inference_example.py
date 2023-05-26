import numpy as np
import pandas as pd
import json
from transformers import *
import os
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pickle
import time
from sklearn.model_selection import train_test_split
### define tokenizer ###
model_name = "monologg/kobigbird-bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

### preprocess data ###
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

### get all candidate input ids and attention masks ###
candidate_input_ids, candidate_attn_masks = [], []
for i in tqdm(range(len(passages)), position=0, leave=True):
    encoded_inputs = tokenizer(str(passages[i]), max_length=512, truncation=True, padding="max_length")
    candidate_input_ids.append(encoded_inputs["input_ids"])
    candidate_attn_masks.append(encoded_inputs["attention_mask"])
candidate_input_ids = torch.tensor(candidate_input_ids, dtype=int)
candidate_attn_masks = torch.tensor(candidate_attn_masks, dtype=int)

### get query-answer lookup dictionary ###
query_answer_index_dict = {}
for i in range(len(queries)):
    query_answer_index_dict[queries[i]] = []
for i in range(len(queries)):
    query_answer_index_dict[queries[i]].append(i)

### train/validation/test split ###
train_df, val_df = train_test_split(all_data, test_size=0.2, random_state=42) # 80% train
val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42) # 10% validation, 10% test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_q_chkpt = torch.load("maum_lawqa_query_encoder.pt")
q_encoder = AutoModel.from_pretrained("monologg/kobigbird-bert-base")
q_encoder.to(device)
q_encoder.load_state_dict(best_q_chkpt)

best_p_chkpt = torch.load("maum_lawqa_passage_encoder.pt")
p_encoder = AutoModel.from_pretrained("monologg/kobigbird-bert-base")
p_encoder.to(device)
p_encoder.load_state_dict(best_p_chkpt)

top5_answers = []
with torch.no_grad():
    p_encoder.eval()
    q_encoder.eval()
    p_embs = []
    inference_dataset = TensorDataset(candidate_input_ids, candidate_attn_masks)
    inference_sampler = SequentialSampler(inference_dataset)
    inference_dataloader = DataLoader(inference_dataset, sampler=inference_sampler, batch_size=64)
    for step, batch in tqdm(enumerate(inference_dataloader), position=0, leave=True, total=len(inference_dataloader), desc="calculating embeddings"):
        batch = (t.to(device) for t in batch)
        b_input_ids, b_attn_masks = batch
        p_emb = p_encoder(b_input_ids, b_attn_masks).pooler_output
        for i in range(p_emb.shape[0]):
            p_embs.append(torch.reshape(p_emb[i], (-1, 768)))
    p_embs = torch.cat(p_embs, dim=0)
    print(f"candidate embeddings shape: {p_embs.shape}")
    top_1, top_5 = 0, 0
    test_questions = test_df["queries"].values
    for sample_idx in tqdm(range(len(test_questions)), position=0, leave=True, desc="Calculating recall"):
        query = test_questions[i]
        encoded_query = tokenizer(str(query), max_length=512, truncation=True, padding="max_length", return_tensors="pt").to(device)
        q_emb = q_encoder(**encoded_query).pooler_output
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        correct_idx = query_answer_index_dict[query]
        cnt = 0
        for idx in correct_idx:
            if idx in rank[:1]:
                cnt += 1
        top_1 += cnt / min(1, len(correct_idx))
        cnt = 0
        for idx in correct_idx:
            if idx in rank[:5]:
                cnt += 1
        top_5 += cnt / min(5, len(correct_idx))
    avg_top_1 = top_1 / len(test_questions)
    avg_top_5 = top_5 / len(test_questions)
    print(f"test recall@1 : {avg_top_1} | test recall@5: {avg_top_5}")

