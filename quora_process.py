import os
import nltk
import pickle
import random
from transformers import BertTokenizer
from modules.utils import loadpkl
from nltk import word_tokenize
import editdistance

text_path = "/path/quora"

def creat_vocab(valid = True,min_count = 2,data_root="data"):
    train_src_file_path = os.path.join(text_path, 'train_src.txt')
    train_trg_file_path = os.path.join(text_path, 'train_trg.txt')
    if valid:
        valid_src_file_path = os.path.join(text_path, 'valid_src.txt')
        valid_trg_file_path = os.path.join(text_path, 'valid_trg.txt')


    word2count = {}

    with open(train_src_file_path,encoding='utf-8') as f_src, open(train_trg_file_path,encoding='utf-8') as f_trg:
        for line in f_src.readlines()+f_trg.readlines():
            words = nltk.word_tokenize(line)
            words = [word.lower() for word in words]
            tags = list(zip(*nltk.pos_tag(words)))[1]

            for word in words:
                if word not in word2count:
                    word2count[word] = 0
                word2count[word]+=1

            for tag in tags:
                if tag not in word2count:
                    word2count[tag] = 0
                word2count[tag]+=1


    if valid:
        with open(valid_src_file_path, encoding='utf-8') as f_src, open(valid_trg_file_path, encoding='utf-8') as f_trg:
            for line in f_src.readlines() + f_trg.readlines():
                words = nltk.word_tokenize(line)
                words = [word.lower() for word in words]
                tags = list(zip(*nltk.pos_tag(words)))[1]

                for word in words:
                    if word not in word2count:
                        word2count[word] = 0
                    word2count[word] += 1

                for tag in tags:
                    if tag not in word2count:
                        word2count[tag] = 0
                    word2count[tag] += 1

    word2idx = {'PAD': 0, 'UNK': 1, 'SOS': 2, 'EOS': 3}
    idx = 4
    for word,count in word2count.items():
        if count >= min_count:
            if word not in word2idx:
                word2idx[word] = idx
                idx += 1

  
    idx2word = {v: k for k, v in word2idx.items()}
    with open(os.path.join(data_root,"word2idx.pkl"),'wb') as f:
        pickle.dump(word2idx,f)
    with open(os.path.join(data_root,"idx2word.pkl"),'wb') as f:
        pickle.dump(idx2word,f)

def token_to_idx(sub_file='/train'):
    word2idx = loadpkl("data/word2idx.pkl")
    src_path = text_path+sub_file+"_src.txt"
    trg_path = text_path+sub_file+"_trg.txt"
    src_idx1=[]
    trg_idx1=[]
    
    src_idx2=[]
    trg_idx2=[]

    with open(src_path,encoding='utf-8') as f:
        for line in f.readlines():
            words = nltk.word_tokenize(line)
            words = [word.lower() for word in words]
            words_idx = [word2idx[word] if word in word2idx else word2idx['UNK'] for word in words]
            tags = list(zip(*nltk.pos_tag(words)))[1]
            tags_idx = [word2idx[word] if word in word2idx else word2idx['UNK'] for word in tags]
            src_idx1.append(tags_idx)
            src_idx2.append(words_idx)

    with open(trg_path,encoding='utf-8') as f:
        for line in f.readlines():
            words = nltk.word_tokenize(line)
            words = [word.lower() for word in words]
            words_idx = [word2idx[word] if word in word2idx else word2idx['UNK'] for word in words]
            tags = list(zip(*nltk.pos_tag(words)))[1]
            tags_idx = [word2idx[word] if word in word2idx else word2idx['UNK'] for word in tags]
            trg_idx1.append(tags_idx)
            trg_idx2.append(words_idx)

    with open("data"+sub_file+"/src_pos.pkl",'wb') as f:
        pickle.dump(src_idx1,f)

    with open("data"+sub_file+"/trg_pos.pkl",'wb') as f:
        pickle.dump(trg_idx1,f)

    with open("data"+sub_file+"/src.pkl",'wb') as f:
        pickle.dump(src_idx2,f)

    with open("data"+sub_file+"/trg.pkl",'wb') as f:
        pickle.dump(trg_idx2,f)
    

def id_to_bertid(sub_file="/test"):


    idx2word = loadpkl("data/idx2word.pkl")
    input_path = "data"+sub_file+"/src.pkl"
    output_path = "data"+sub_file+"/trg.pkl"

    input_s = loadpkl(input_path)
    output_s = loadpkl(output_path)


    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_src_text = [' '.join([idx2word[word] for word in sent]) for sent in input_s]
    bert_trg_text = [' '.join([idx2word[word] for word in sent]) for sent in output_s]
    bert_src = [tokenizer.encode(sent, add_special_tokens=True) for sent in bert_src_text]
    bert_trg = [tokenizer.encode(sent, add_special_tokens=True) for sent in bert_trg_text]

    with open("data"+sub_file+"/bert_src.pkl",'wb') as f:
        pickle.dump(bert_src,f)
    with open("data"+sub_file+"/bert_trg.pkl",'wb') as f:
        pickle.dump(bert_trg,f)

def find_exm(sub_file="/test"):

    tags = loadpkl("data"+sub_file+"/trg_pos.pkl")
    trgs = loadpkl("data"+sub_file+"/trg.pkl")

    similar_list = []
    for i in range(len(trgs)):
        len_i = len(trgs[i])
        same_number = [100 for k in range(len(trgs))] 

        if i % 1000 == 0:
            print(i)
        for j in range(len(trgs)):
            if i != j:
                len_j = len(trgs[j])
                if abs(len_i - len_j) > 2:
                    continue
                if len(list(set(trgs[i]) & set(trgs[j]))) + 2 > len(list(set(trgs[i]))):
                    continue
                syn_tags = tags[i]
                temp_tags = tags[j]
                posed = editdistance.eval(syn_tags, temp_tags)
                same_number[j] = posed


        m = min(same_number)
        aaa = [i for i, j in enumerate(same_number) if j == m]

        similar_list.append(random.sample(aaa, min(5, len(aaa))))

    with open("data"+sub_file+"/sim.pkl", 'wb') as f:
        pickle.dump(similar_list, f)
    print(similar_list)

#execute below function one by one

#creat_vocab()
#token_to_idx(sub_file='/train')
#token_to_idx(sub_file='/valid')
#token_to_idx(sub_file='/test')
#find_exm("/train")
#find_exm("/valid")
#find_exm("/test")
#id_to_bertid("/train")
#id_to_bertid("/valid")
#id_to_bertid("/test")
