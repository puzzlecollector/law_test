import grpc
import argparse
from myservice_pb2 import MyInput
from myservice_pb2_grpc import MyServiceNameStub

class MyServiceNameClient:
    def __init__(self, ip="localhost", port=35015):
        self.server_ip = ip
        self.server_port = port
        self.stub = MyServiceNameStub(
            grpc.insecure_channel(self.server_ip + ":" + str(self.server_port))
        )

    def get_myservice(self, query_str, candidate_path, query_index_dict_path, law_large_df_path):
        myinput = MyInput()
        myinput.query_str = query_str
        myinput.candidate_path = candidat_path
        myinput.query_index_dict_path = query_index_dict_path
        myinput.law_large_df_path = law_large_df_path
        myservice_out = self.stub.processor(myinput)
        return myservice_out.mystrout

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-query", type=str)
    parser.add_arugment("-candidate", type=str)
    parser.add_argument("-query_index_dict_path", type=str)
    parser.add_argument("-law_large_df_path", type=str)
    parser.add_argument("--ip", default="localhost", type=str)
    parser.add_argument("--port", default=35015, type=int)
    args = parser.parse_args()
    myservice_client = MyServiceNameClient(ip=args.ip, port=args.port)
    print("Ouptut : ", myservice_client.get_myservice(query_str = args.query,
                                                      candidate_path = args.candidate,
                                                      query_index_dict_path=args.query_index_dict_path,
                                                      law_large_df_path = args.law_large_df_path))
    print("done!")
    print("="*100) 
