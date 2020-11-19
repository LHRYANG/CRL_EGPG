from modules.Seq2Seq_Module import Encoder, Decoder
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import pickle
import logging
import os
from modules.utils import initialise_word_embedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger('Seq2Seq')
logging.basicConfig()
handle = logging.StreamHandler()
handle.setLevel(logging.DEBUG)
handle.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handle)
logger.setLevel(logging.DEBUG)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Seq2Seq(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        embedding_dim = config_dict.get('embedding_dim', 1)
        vocabulary_dim = config_dict.get('vocabulary_dim', 1)
        self.emb_layer = nn.Embedding(num_embeddings=vocabulary_dim, embedding_dim=embedding_dim, padding_idx=0)
        logger.info('Initializing word vectors from pretrained model: glove....')
        glove_weight = initialise_word_embedding()
        self.emb_layer.weight.data.copy_(torch.from_numpy(glove_weight))
        logger.info('Word Vectors initialized')

        self.encoder_layer = Encoder(config_dict=config_dict)
        self.decoder = Decoder(config_dict)
        self.decoder.word_emb_layer = self.emb_layer
        self.para = list(filter(lambda x: x.requires_grad, self.parameters()))
        self.opt = Adam(params=self.para, lr=config_dict.get('lr', 1e-4))

    def forward(self, seq_arr, seq_len,style_emb, response=None, decoder_input=None, max_seq_len=16):

        encode_mask = (seq_arr == 0).byte()
        seq_arr = self.emb_layer(seq_arr)
        encode_output, encode_hidden = self.encoder_layer(seq_arr, seq_len)
        all_output = self.decoder(encode_hidden, encode_output, encode_mask, response,
                                  decoder_input,style_emb,max_seq_len=max_seq_len)
        return all_output,encode_hidden





   