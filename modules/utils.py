import pickle
import numpy as np

def writepkl(path,obj):
    with open(path,'wb') as f:
        pickle.dump(obj,f)

def loadpkl(path):
    with open(path,'rb') as f:
        obj = pickle.load(f)
    return obj

def getword(idx2word,lists):
    results = []
    for list in lists:
        append = False
        result = []
        for idx in list:
            if idx == 0 or idx == 2:
                continue
            if idx == 3:
                append = True
                results.append(result)
                break
            result.append(idx2word[idx.item()])
        if not append:
            results.append(result)
    return results

def print_line_by_line(arr):
    for a in arr:
        print(a)
    print("-----------------------")

def load_glove_model(File):
    print("Loading Glove Model")
    f = open(File,'r')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    print(len(gloveModel)," words loaded!")
    return gloveModel

def initialise_word_embedding(File="/gds/hryang/projdata11/hryang/data/visual_question_answering/glove/glove.6B.300d.txt"):
    with open("./data2/word2idx.pkl",'rb') as f:
        vocab = pickle.load(f)
    glove_emb = load_glove_model(File)
    word_emb = np.zeros((len(vocab),300))
    miss_num = 0
    for word,idx in vocab.items():
        if word in glove_emb:
            word_emb[idx] = glove_emb[word]
            continue
        miss_num+=1
    print(str(miss_num)+" words are not in the glove embedding")
    return word_emb