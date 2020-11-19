import json
import pickle
import torch
from modules.Seq2Seq2 import Seq2Seq
from modules.StyleExtractor import StyleExtractor
from modules.datasets import STdata
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
import argparse
import nltk

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--dataset', type=str, default='para',
                        choices=['para', 'quora'], help='dataset')
    parser.add_argument('--max_len', type=int, default=15,
                        help='max_len of sentence')
    parser.add_argument('--model_save_path', type=str, default="save_model/ours_para")
    parser.add_argument('--idx', type=int, default=30)
    parser.add_argument('--idx2', type=int, default=4)

    opt = parser.parse_args()

    if opt.dataset == 'quora':
        opt.data_folder = './data/'
        opt.config="config.json"
    elif opt.dataset == 'para':
        opt.data_folder = './data2/'
        opt.config="config2.json"
    return opt


opt = parse_option()
with open(opt.config) as f:
    config = json.load(f)

seq2seq = Seq2Seq(config).to(device)
stex = StyleExtractor(config).to(device)


seq2seq.load_state_dict(torch.load(os.path.join(opt.model_save_path,"seq2seq"+str(opt.idx)+".pkl")))
stex.load_state_dict(torch.load(os.path.join(opt.model_save_path,"stex"+str(opt.idx)+".pkl")))

seq2seq.eval()
stex.eval()
seq2seq.decoder.mode = "infer"

with open(os.path.join(opt.data_folder,'idx2word.pkl'), 'rb') as f:
    idx2word = pickle.load(f)
with open(os.path.join(opt.data_folder,'word2idx.pkl'), 'rb') as f:
    word2idx = pickle.load(f)
with open(os.path.join(opt.data_folder,'test/bert_src.pkl'),'rb') as f:
    bert_output = pickle.load(f)
with open(os.path.join(opt.data_folder,'test/trg.pkl'),'rb') as f:
    normal_output = pickle.load(f)


src = ["how do i develop good project management skills ?"]
#src = ["which is the best anime to watch ?"]
#src = ["if you are at the lowest point of your life , what do you do ?"]
#src = ["when did you first realize that you were gay lesbian bi ?"]

#src = ["he believed his son had died in a terrorist attack ."]
#src = ["it is hard for me to imagine where they could be hiding it underground ."]
#src = ["do you want to kiss teddy ?"]
#src = ["i had a strange call from a woman ."]
def pack_sim(sim):
    bert_sim = torch.zeros(opt.max_len + 2, dtype=torch.long)
    bert_sim[0:min(len(sim), opt.max_len + 2)] = torch.tensor(sim[0:min(opt.max_len + 2, len(sim))])
    return bert_sim
def pack_input(src):
    src_tokennized = [nltk.word_tokenize(line) for line in src]
    src_idx = [[word2idx[word] if word in word2idx else word2idx['UNK'] for word in words] for words in src_tokennized]
    src_idx = src_idx[0]
    in_len = len(src_idx)
    src_out = torch.zeros(opt.max_len, dtype=torch.long)

    if in_len > opt.max_len:
        src_out[0:opt.max_len] = torch.tensor(src_idx[0:opt.max_len])
        in_len = opt.max_len
    else:
        src_out[0:in_len] = torch.tensor(src_idx)

    src_out = src_out.unsqueeze(0)
    length = torch.tensor([in_len])
    return src_out, length

src_idx, in_len = pack_input(src)


with torch.no_grad():
    count = 0
    temp_arr = []
    src_idx = src_idx.to(device)
    in_len = in_len.to(device)
    for style in bert_output:
        bert_sim = pack_sim(style)
        bert_sim = bert_sim.unsqueeze(0)
        bert_sim = bert_sim.to(device)
        style_emb = stex(bert_sim)
        id_arr,_ = seq2seq.forward(src_idx, in_len, style_emb)
        temp_arr.append(id_arr)
       


    filename = os.path.join(opt.model_save_path,"gen"+str(opt.idx2)+".txt")
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'a') as f:
        for bat in temp_arr:
            ss = seq2seq.getword(idx2word, bat)
            for s in ss:
                f.write(' '.join(s))
                f.write('\n')

