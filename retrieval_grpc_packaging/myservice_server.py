import time
import grpc
from concurrent import futures
import argparse
import torch
import torch.nn as nn
from myservice_pb2 import MyOutput
from myservice_pb2_grpc import MyServiceNameServicer, add_MyServiceNameServicer_to_server
from utils import TextEncoder, CreateIndices, TextRetriever, Trainer
import numpy as np
import pandas as pd
import os
import pickle

class MyServiceNameServer(MyServiceNameServicer):
    def __init__(self, plm="monologg/kobigbird-bert-base", topK = 50):
        self.plm = plm
        self.topK = topK
        self.best_q_chkpt = torch.load("large_law_KoBigBird_query_encoder.pt")
        self.best_p_chkpt = torch.load("large_law_KoBigBird_passage_encoder.pt")

        self.q_encoder = AutoModel.from_pretrained(self.plm)
        self.q_encoder.to(self.device)
        self.q_encoder.load_state_dict(self.best_q_chkpt)

        self.p_encoder = AutoModel.from_pretrained(self.plm)
        self.p_encoder.to(self.device)
        self.p_encoder.load_state_dict(self.best_p_chkpt)

        self.q_tokenizer = AutoTokenizer.from_pretrained(self.plm)
        self.p_tokenizer = AutoTokenizer.from_pretrained(self.plm)

    def processor(self, request, context):
        # load data and model checkpoints
        if request.query_str is not None:
            print(f"query string : {request.query_str}")
            query = request.query_str
        if request.candidate_path is not None:
            p_embs = torch.load(request.candidate_path)
        if request.query_index_dict_path is not None:
            with open(request.query_index_dict_path, "rb") as f:
                query_index_dict = pickle.load(f)
        if request.law_large_df_path is not None:
            all_data = pd.read_csv(request.law_large_df_path)
            candidate_texts = all_data["candidates"].values
        # inference
        with torch.no_grad():
            self.p_encoder.eval()
            self.q_encoder.eval()

            encoded_query = self.q_tokenizer(str(query), max_length=512, truncation=True, padding="max_length", return_tensors="pt").to(self.device)
            q_emb = self.q_encoder(**encoded_query).pooler_output
            dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
            rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
            print(f"가장 적합한 Top {self.topK}개의 판결요지 가져오는중")
            rank = rank[:self.topK]

            ground_truth_idxs = query_index_dict[query]
            ranked_positions = []

            for i in range(len(self.topK)):
                print(f"============== Rank {i+1} ==============")
                print(candidate_texts[rank[i]])
                if rank[i] in ground_truth_idxs:
                    ranked_positions.append(i+1)

            exists = False
            for idx in ground_truth_idxs:
                if idx in rank:
                    exists = True

            print(f"실제 정답 판결요지가 랭크된 top {self.topK}에 존재하나요?: {"네" if exists else "아니요"}")
            print(f"실제 정답 판결요지의 랭크 위치: {str(ranked_positions[0])}")

            return MyOutput(mystrout = candidate_texts[rank[0]]) # return the highest ranked text

if __name__ == "__main__":
    port = 35015
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1),)
    my_service_server = MyServiceNameServer()
    my_service_servicer = add_MyServiceNameServicer_to_server(my_service_server, server)
    server.add_insecure_port("[::]:{}".format(port))
    server.start()
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        server.stop(0)
