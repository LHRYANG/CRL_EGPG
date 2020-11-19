from torch.utils.data import DataLoader
import torch
import json
from tqdm import tqdm
from modules.datasets import STdata
from modules.Seq2Seq import Seq2Seq
from modules.StyleExtractor import StyleExtractor
from modules.loss import SupConLoss
from torch.optim import Adam
import warnings
import numpy as np
import pickle
import os
import torch.nn.functional as F
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--save_freq', type=int, default=15,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=45,
                        help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--dataset', type=str, default='quora',
                        choices=['para', 'quora'], help='dataset')
    parser.add_argument('--max_len', type=int, default=15,
                        help='max_len of sentence')
    parser.add_argument('--ckpt', type=str, default='100_512_1000.pth',
                        help='path to pre-trained model')
    parser.add_argument('--model_save_path', type=str, default="save_model/ours_quora")
    parser.add_argument('--infer_save_path', type=str, default="text.txt")

    opt = parser.parse_args()

    if opt.dataset == 'quora':
        opt.data_folder = './data/'
        opt.config="config.json"
    elif opt.dataset == 'para':
        opt.data_folder = './data2/'
        opt.config="config2.json"
    return opt


def train_epoch(seq2seq,stex,optimizer,dataloader, epoch):
    criterion = SupConLoss()
    seq2seq.train()
    seq2seq.decoder.mode = "train"
    stex.train()

    loss_arr = []
    closs_arr = []
    sloss_arr = []
    count = 0
    for src,in_len,trg,trg_input,ou_len,bert_src,bert_trg,bert_sim,content_trg,content_len in dataloader:
        optimizer.zero_grad()
        src,in_len, trg, trg_input, ou_len, bert_src, bert_trg, bert_sim,content_trg,content_len=\
            src.to(device),in_len.to(device),trg.to(device),trg_input.to(device),ou_len.to(device)\
                ,bert_src.to(device),bert_trg.to(device),bert_sim.to(device),content_trg.to(device),content_len.to(device)

        style_emb = stex(bert_trg)
        style_emb2 = stex(bert_sim)
        nll_loss2,src_hidden = seq2seq.forward(src, in_len, style_emb2,response=trg, decoder_input=trg_input)
        
        seq_arr = seq2seq.emb_layer(content_trg)
        _,trg_hidden = seq2seq.encoder_layer(seq_arr, content_len)
       
        content_src = torch.unsqueeze(F.normalize(src_hidden,dim=1),1)
        content_trg = torch.unsqueeze(F.normalize(trg_hidden,dim=1),1)
        content_contrast = torch.cat((content_src,content_trg),dim=1)

        style_trg = style_emb[:, -1, :]
        style_exm = style_emb2[:, -1,:]
        style_trg = torch.unsqueeze(F.normalize(style_trg,dim=1),1)
        style_exm = torch.unsqueeze(F.normalize(style_exm,dim=1),1)
        style_contrast = torch.cat((style_trg,style_exm),dim=1)

        content_loss = criterion(content_contrast)
        style_loss = criterion(style_contrast)

        nll_loss = nll_loss2 + 0.1*(content_loss+style_loss)#nll_loss1 + nll_loss2 + nll_loss3
        nll_loss.backward()
        optimizer.step()
        loss_arr.append(nll_loss2.cpu().item())
        closs_arr.append(content_loss.cpu().item())
        sloss_arr.append(style_loss.cpu().item())

    loss1 = np.mean(loss_arr)
    loss2 = np.mean(closs_arr)
    loss3 = np.mean(sloss_arr)
    print("epoch: ", epoch,loss1,loss2,loss3)
    return [loss1,loss2,loss3]
def eval_epoch(seq2seq,stex,dataloader, epoch):
    seq2seq.eval()
    seq2seq.decoder.mode = "eval"
    stex.eval()

    ppl_arr = []
    nll_arr = []
    with torch.no_grad():
        for src, in_len, trg, trg_input, ou_len, bert_src, bert_trg, bert_sim,content_trg,content_len in dataloader:
            src, in_len, trg, trg_input, ou_len, bert_src, bert_trg, bert_sim,content_trg,content_len = \
                src.to(device), in_len.to(device), trg.to(device), trg_input.to(device), ou_len.to(device) \
                    , bert_src.to(device), bert_trg.to(device), bert_sim.to(device),content_trg.to(device),content_len.to(device)

            style_emb2 = stex(bert_sim)
            total_output,_ = seq2seq.forward(src, in_len, style_emb2, response=trg, decoder_input=trg_input)
            ppl2 ,nll2 = total_output
            ppl_arr.append(ppl2)
            nll_arr.append(nll2.cpu().item())
        ppl = torch.cat(ppl_arr, dim=0)
        ppl_mask = (ppl < 200).float()
        ppl = (ppl * ppl_mask).sum() / ppl_mask.sum()
        ppl = ppl.cpu().item()
        nll = np.mean(nll_arr)
        print(nll,ppl)
    return [nll,ppl]

if __name__ == '__main__':
    opt = parse_option()
    with open(opt.config) as f:
        config = json.load(f)
    train_set = STdata("train",dataroot=opt.data_folder,max_len=opt.max_len)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)

    valid_set = STdata("valid",dataroot=opt.data_folder,max_len=opt.max_len)
    valid_loader = DataLoader(valid_set, batch_size=opt.batch_size, shuffle=False)

    test_set = STdata("test",dataroot=opt.data_folder,max_len=opt.max_len)
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False)

    seq2seq = Seq2Seq(config).to(device)
    stex = StyleExtractor(config).to(device)

    params = list(seq2seq.parameters()) + list(stex.parameters())
    optimizer = Adam(params,lr=opt.lr)


    if not os.path.exists(opt.model_save_path):
        os.makedirs(opt.model_save_path)

    loss_arr=[]
    ppl_arr=[]
    for epoch in range(opt.epochs):
        loss = train_epoch(seq2seq,stex,optimizer,train_loader, epoch)
        ppl = eval_epoch(seq2seq,stex,valid_loader, epoch)
        loss_arr.append(loss)
        ppl_arr.append(ppl)
        if (epoch+1)%opt.save_freq==0:
            torch.save(seq2seq.state_dict(), os.path.join(opt.model_save_path,'seq2seq'+str(epoch+1) + '.pkl'))
            torch.save(stex.state_dict(), os.path.join(opt.model_save_path,'stex'+str(epoch+1) + '.pkl'))

    with open(os.path.join(opt.model_save_path,'loss.pkl'),'wb') as f:
        pickle.dump(loss_arr,f)

    with open(os.path.join(opt.model_save_path,'ppl.pkl'),'wb') as f:
        pickle.dump(ppl_arr,f)

