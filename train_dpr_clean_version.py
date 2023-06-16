import json
import numpy as np 
import pandas as pd
from pytorch_metric_learning import miners, losses, distances
import os 
from transformers import * 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler 
import pickle 
import time 
from sklearn.model_selection import train_test_split, KFold
from tqdm.auto import tqdm 
import random 
import math 
import faiss

def read_jsonl_file(file_path):
    arr = [] 
    with open(file_path, 'r') as file:
        for line in file:
            json_data = json.loads(line)
            arr.append(json_data)
    return arr

data = [] 
with open("law_candidates_60K.jsonl") as f: 
	for line in f: 
		data.append(json.loads(line))
queries, passages = [], []
for i in tqdm(range(len(data)), position=0, leave=True):
	query = data[i]["summary"] 
	passage = data[i]["text"] 
	queries.append(query) 
	passages.append(passage) 

other_data = [] 
with open("legalqa_1.8K.jsonlines") as f: 
	for line in f: 
		other_data.append(json.loads(line)) 
for i in tqdm(range(len(other_data)), position=0, leave=True): 
	query = other_data[i]["question"] 
	passage = other_data[i]["answer"] 
	queries.append(query) 
	passages.append(passage) 

df = pd.DataFrame(list(zip(queries, passages)), columns=["queries", "passages"]) 
df = df.sample(frac=1).reset_index(drop=True) 
# udpate queries and passages after shuffling indices 
queries = df["queries"].values 
passages = df["passages"].values 

model_name = "monologg/kobigbird-bert-base" 
q_tokenizer = AutoTokenizer.from_pretrained(model_name) 
p_tokenizer = AutoTokenizer.from_pretrained(model_name) 

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42) 
valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)  

query_index_dict = {} 
for i in range(len(queries)): 
	query_index_dict[queries[i]] = [] 
for i in range(len(queries)): 
	query_index_dict[queries[i]].append(i) # query string: list of relevant candidate ids 

candidates = df["passages"].values 
all_context_input_ids, all_context_attn_masks = [], []
for i in tqdm(range(len(candidates)), position=0, leave=True): 
	encoded_inputs = p_tokenizer(str(candidates[i]), max_length=512, truncation=True, padding="max_length") 
	all_context_input_ids.append(encoded_inputs["input_ids"]) 
	all_context_attn_masks.append(encoded_inputs["attention_mask"]) 
all_context_input_ids = torch.tensor(all_context_input_ids, dtype=int) 
all_context_attn_masks = torch.tensor(all_context_attn_masks, dtype=int) 

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
		self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
		self.chunk_size = 512 # we are not dealing with very long texts 
	def __call__(self, batch): 
		q_input_ids, q_attn_masks, q_labels = [], [], []
		p_input_ids, p_attn_masks, p_labels = [], [], [] 
		ids = 0 
		for idx, row in enumerate(batch): 
			query, passage = row[0], row[1]
			encoded_q = self.tokenizer(query, max_length=self.chunk_size, padding="max_length", truncation=True, return_tensors="pt") 
			encoded_p = self.tokenizer(passage, max_length=self.chunk_size, padding="max_length", truncation=True, return_tensors="pt") 
			q_input_ids.append(encoded_q["input_ids"]) 
			q_attn_masks.append(encoded_q["attention_mask"]) 
			q_labels.append(ids) 
			p_input_ids.append(encoded_p["input_ids"]) 
			p_attn_masks.append(encoded_p["attention_mask"]) 
			p_labels.append(ids) 
			ids += 1 
		q_input_ids = torch.stack(q_input_ids, dim=0).squeeze(dim=1) 
		q_attn_masks = torch.stack(q_attn_masks, dim=0).squeeze(dim=1) 
		q_labels = torch.tensor(q_labels, dtype=int) 
		p_input_ids = torch.stack(p_input_ids, dim=0).squeeze(dim=1) 
		p_attn_masks = torch.stack(p_attn_masks, dim=0).squeeze(dim=1) 
		p_labels = torch.tensor(p_labels, dtype=int) 
		return q_input_ids, q_attn_masks, q_labels, p_input_ids, p_attn_masks, p_labels 

train_set = PairData(train_df) 
valid_set = PairData(valid_df) 
collate = custom_collate() 
train_dataloader = DataLoader(train_set, batch_size=32, collate_fn=collate, shuffle=True) 
valid_dataloader = DataLoader(valid_set, batch_size=32, collate_fn=collate, shuffle=False) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# define model structure
class MeanPooling(nn.Module): 
	def __init__(self): 
		super(MeanPooling, self).__init__() 
	def forward(self, last_hidden_state, attention_masks): 
		input_mask_expanded = attention_masks.unsqueeze(-1).expand(last_hidden_state.size()).float() 
		sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1) 
		sum_mask = input_mask_expanded.sum(1) 
		sum_mask = torch.clamp(sum_mask, min=1e-9) 
		mean_embeddings = sum_embeddings / sum_mask 
		return mean_embeddings

class BertEmbedder(nn.Module): 
	def __init__(self, plm):
		super(BertEmbedder, self).__init__() 
		self.encoder = AutoModel.from_pretrained(plm)
		self.mean_pooler = MeanPooling() 
	def forward(self, input_ids, attn_masks): 
		x = self.encoder(input_ids, attn_masks)[0] 
		x = self.mean_pooler(x, attn_masks) 
		return x 
	
q_encoder = BertEmbedder(plm=model_name) 
q_encoder.to(device) 
p_encoder = BertEmbedder(plm=model_name) 
p_encoder.to(device) 

# train with the help of  torch metric learning library 
epochs = 30 
params = list(q_encoder.parameters()) + list(p_encoder.parameters())
optimizer = torch.optim.AdamW(params, lr=2e-5, eps=1e-8) # not tuned 
t_total = len(train_dataloader) * epochs 
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * t_total), num_training_steps=t_total) 
miner = miners.MultiSimilarityMiner() # online batch mining for hard negatives 
loss_func = losses.MultiSimilarityLoss() # alternatives: contrastive loss, infoNCE loss, triplet margin loss 
q_encoder.zero_grad() 
p_encoder.zero_grad()
torch.cuda.empty_cache() 
best_recall = 0 
train_losses, val_losses = [], [] 
val_recall_1, val_recall_5 = [], [] # focus on R@1, R@5 
for epoch_i in tqdm(range(0, epochs), desc="Epochs", position=0, leave=True, total=epochs): 
	train_loss = 0
	q_encoder.train() 
	p_encoder.train() 
	with tqdm(train_dataloader, unit="batch") as tepoch: 
		for step, batch in enumerate(tepoch):
			batch = tuple(t.to(device) for t in batch) 
			q_input_ids, q_attn_masks, q_labels, p_input_ids, p_attn_masks, p_labels = batch 
			q_embeddings = q_encoder(q_input_ids, q_attn_masks) 
			p_embeddings = p_encoder(p_input_ids, p_attn_masks) 
			full_embeddings = torch.cat((q_embeddings, p_embeddings), dim=0)
			full_labels = torch.cat((q_labels, p_labels))
			hard_pairs = miner(full_embeddings, full_labels)
			loss = loss_func(full_embeddings, full_labels, hard_pairs) 
			train_loss += loss.item() 
			loss.backward() 
			torch.nn.utils.clip_grad_norm_(params, 1.0) 
			optimizer.step()
			scheduler.step() 
			q_encoder.zero_grad() 
			p_encoder.zero_grad() 
			tepoch.set_postfix(loss=train_loss/(step+1)) 
			time.sleep(0.1) 
	avg_train_loss = train_loss / len(train_dataloader)
	val_loss = 0 
	q_encoder.eval() 
	p_encoder.eval() 
	for step, batch in tqdm(enumerate(valid_dataloader), position=0, leave=True, total=len(valid_dataloader)): 
		batch = tuple(t.to(device) for t in batch) 
		q_input_ids, q_attn_masks, q_labels, p_input_ids, p_attn_masks, p_labels = batch 
		with torch.no_grad(): 
			q_embeddings = q_encoder(q_input_ids, q_attn_masks) 
			p_embeddings = p_encoder(p_input_ids, p_attn_masks) 
			full_embeddings = torch.cat((q_embeddings, p_embeddings), dim=0) 
			full_labels = torch.cat((q_labels, p_labels)) 
			loss = loss_func(full_embeddings, full_labels) 
			val_loss += loss.item() 
	avg_val_loss = val_loss / len(valid_dataloader) 
	train_losses.append(avg_train_loss) 
	val_losses.append(avg_val_loss) 

	# calculate recall metrics using faiss cosine similarity 
	with torch.no_grad(): 
		recall = 0 
		p_encoder.eval() 
		p_embs = [] 
		inference_dataset = TensorDataset(all_context_input_ids, all_context_attn_masks) 
		inference_sampler = SequentialSampler(inference_dataset) 
		inference_dataloader = DataLoader(inference_dataset, sampler=inference_sampler, batch_size=64) 
		for step, batch in tqdm(enumerate(inference_dataloader), position=0, leave=True, total=len(inference_dataloader), desc="calculating candidate embeddings with current model weight"): 
			batch = (t.to(device) for t in batch) 
			b_input_ids, b_attn_masks = batch 
			p_emb = p_encoder(b_input_ids, b_attn_masks) 
			p_emb = p_emb.detach().cpu() 
			for i in range(p_emb.shape[0]): 
				p_embs.append(torch.reshape(p_emb[i], (-1, 768)))
		p_embs = torch.cat(p_embs, dim=0) 
		p_embs = p_embs.detach().cpu().numpy() 
		print(f"==== sanity check: candidate embeddings shape: {p_embs.shape} ===")
		index = faiss.IndexIDMap2(faiss.IndexFlatIP(768))
		faiss.normalize_L2(p_embs.astype(np.float32)) 
		index.add_with_ids(p_embs.astype(np.float32), np.array(range(0, len(p_embs)), dtype=int)) 
		index.nprobe = 64
		top_1, top_5 = 0, 0
		q_encoder.eval() 
		val_questions = valid_df["queries"].values 
		for sample_idx in tqdm(range(len(val_questions)), position=0, leave=True, desc="Calculating Recall"): 
			query = val_questions[sample_idx]
			encoded_query = q_tokenizer(str(query), max_length=512, truncation=True, padding="max_length", return_tensors="pt").to(device) 
			input_id = encoded_query["input_ids"] 
			attn_mask = encoded_query["attention_mask"] 
			q_emb = q_encoder(input_id, attn_mask) 
			q_emb = q_emb.detach().cpu().numpy() 
			distances, indices = index.search(q_emb, 1000) # not tuned but results should be somewhat consistent 
			correct_idx = query_index_dict[query] 
			cnt = 0 
			for idx in correct_idx: 
				if idx in indices[0][:1]:
					cnt += 1 
			top_1 += cnt / min(1, len(correct_idx)) 

			cnt = 0 
			for idx in correct_idx: 
				if idx in indices[0][:5]: 
					cnt += 1 
			top_5 += cnt / min(5, len(correct_idx)) 

		avg_top_1 = top_1 / len(val_questions) 
		avg_top_5 = top_5 / len(val_questions) 

		val_recall_1.append(avg_top_1) 
		val_recall_5.append(avg_top_5) 

		print(f"avg train loss : {avg_train_loss} | avg val loss : {avg_val_loss}")
		print(f"avg R@1: {avg_top_1} | avg R@5: {avg_top_5}") 
		
		if avg_top_1 > best_recall: 
			# save based on R@1 metric 
			best_recall = avg_top_1 
			torch.save(q_encoder.state_dict(), "KR_lawqa_KoBigBird_query_encoder.pt") 
			torch.save(p_encoder.state_dict(), "KR_lawqa_KoBigBird_passage_encoder.pt") 
			print("saved current best checkpoints!") 

# save statistics 
with open("train_losses.pkl", "wb") as f: 
	pickle.dump(train_losses, f) 
with open("val_losses.pkl", "wb") as f: 
	pickle.dump(val_losses, f) 
with open("val_recall_1.pkl", "wb") as f: 
	pickle.dump(val_recall_1, f) 
with open("val_recall_5.pkl", "wb") as f: 
	pickle.dump(val_recall_5, f) 
print("done!") 
