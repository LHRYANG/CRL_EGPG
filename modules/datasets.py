from torch.utils.data import Dataset
from modules.utils import loadpkl
import os
import torch
from transformers import BertTokenizer
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class STdata(Dataset):
    def __init__(self,name,dataroot="data",max_len=15):
        super(STdata,self).__init__()
        assert name in ['train','valid','test']
        self.name = name
        self.word2idx = loadpkl(dataroot+'/word2idx.pkl')
        self.idx2word = loadpkl(dataroot+'/idx2word.pkl')

        self.vocab_size = len(self.word2idx)

        input_path = os.path.join(dataroot, name+'/src.pkl')
        output_path = os.path.join(dataroot,name+'/trg.pkl')
        sim_path = os.path.join(dataroot,name+'/sim.pkl')

        bert_input_path = os.path.join(dataroot,name+'/bert_src.pkl')
        bert_output_path = os.path.join(dataroot,name+'/bert_trg.pkl')

        self.input_s = loadpkl(input_path)
        self.output_s = loadpkl(output_path)
        self.sim_s = loadpkl(sim_path)
        self.bert_input = loadpkl(bert_input_path)
        self.bert_output = loadpkl(bert_output_path)

        self.max_len = max_len

    def __getitem__(self,index):
        # seq2seq data_quora format
        input_s = self.input_s[index]
        output_s = self.output_s[index]

        in_len = len(input_s)
        ou_len = len(output_s)

        src = torch.zeros(self.max_len, dtype=torch.long)
        content_trg = torch.zeros(self.max_len,dtype=torch.long)
        content_len = 0

        trg = torch.zeros(self.max_len+1, dtype=torch.long)
        trg_input = torch.zeros(self.max_len+1,dtype=torch.long)

        if in_len>self.max_len:
            src[0:self.max_len] = torch.tensor(input_s[0:self.max_len])
            in_len = self.max_len
        else:
            src[0:in_len] = torch.tensor(input_s)

        if ou_len>self.max_len:
            content_trg[0:self.max_len] = torch.tensor(output_s[0:self.max_len])
            content_len = self.max_len
        else:
            content_trg[0:ou_len] = torch.tensor(output_s)
            content_len = ou_len

        if ou_len>self.max_len:
            trg[0:self.max_len] = torch.tensor(output_s[0:self.max_len])
            trg[self.max_len] = 3 #EOS
            trg_input[1:self.max_len+1] = torch.tensor(output_s[0:self.max_len])
            trg_input[0] = 2
            ou_len = self.max_len+1
        else:
            trg[0:ou_len] = torch.tensor(output_s)
            trg[ou_len] = 3  # EOS
            trg_input[1:ou_len+1] = torch.tensor(output_s)
            trg_input[0] = 2
            ou_len = ou_len+1

        # bert data_quora format
        bert_in = self.bert_input[index]
        bert_out = self.bert_output[index]
        sim = self.bert_output[random.choice(self.sim_s[index])]
        bert_src = torch.zeros(self.max_len+2, dtype=torch.long)
        bert_trg = torch.zeros(self.max_len+2, dtype=torch.long)
        bert_sim = torch.zeros(self.max_len+2, dtype=torch.long)

        bert_src[0:min(len(bert_in),self.max_len+2)] = torch.tensor(bert_in[0:min(self.max_len+2,len(bert_in))])
        bert_trg[0:min(len(bert_out),self.max_len+2)] = torch.tensor(bert_out[0:min(self.max_len + 2, len(bert_out))])
        bert_sim[0:min(len(sim), self.max_len + 2)] = torch.tensor(sim[0:min(self.max_len + 2, len(sim))])

        return src,in_len,trg,trg_input,ou_len,bert_src,bert_trg,bert_sim,content_trg,content_len

    def __len__(self):
        return len(self.input_s)

