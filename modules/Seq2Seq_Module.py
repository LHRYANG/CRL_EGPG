import torch
import torch.nn as nn
from modules.loss import get_nll_loss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        hidden_dim = config_dict['encoder'].get('hidden_dim', 256)
        input_dim = config_dict['encoder'].get('input_dim', 256)
        num_layers = config_dict['encoder'].get('num_layers', 1)
        drop_out = config_dict.get('drop_out', 0.2)
        bidirectional = bool(config_dict['encoder'].get('bidirectional', True))
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, bias=True,
                          batch_first=True, dropout=(0 if num_layers == 1 else drop_out), bidirectional=bidirectional)

    def forward(self, seq_arr, seq_len):

        seq_len_sorted, index = seq_len.sort(dim=-1, descending=True)
        seq_arr_sorted = seq_arr.index_select(0, index)
        padded_input = pack_padded_sequence(seq_arr_sorted, seq_len_sorted, batch_first=True)
        output, hidden = self.rnn(padded_input)
        output, _ = pad_packed_sequence(output, batch_first=True)
        hidden = torch.cat(tuple(hidden), dim=-1)
        _, inverse_index = index.sort(dim=-1, descending=False)
        output = output.index_select(0, inverse_index)
        hidden = hidden.index_select(0, inverse_index)

        return output, hidden


class ScaleDotAttention(nn.Module):
    def __init__(self, config_dict, mode='Dot'):
        super().__init__()
        dim_q = config_dict['decoder'].get('hidden_dim', 256)
        dim_k = config_dict['encoder'].get('final_out_dim', 256)
        if mode == 'Self':
            dim_q = dim_k
        self.W = nn.Parameter(torch.empty([dim_q, dim_k], device=device, requires_grad=True), requires_grad=True)
        nn.init.normal_(self.W, mean=0., std=np.sqrt(2. / (dim_q + dim_k)))

    def forward(self, q, k, v, mask, bias=None):
        attn_weight = k.bmm(q.mm(self.W).unsqueeze(dim=2)).squeeze(dim=2)
        if bias is not None:
            attn_weight += bias

        mask = mask[:,:attn_weight.shape[-1]]
        attn_weight.masked_fill(mask, - float('inf'))
        attn_weight = attn_weight.softmax(dim=-1)
        attn_out = (attn_weight.unsqueeze(dim=2) * v).sum(dim=1)
        return attn_out, attn_weight

class Attention(nn.Module):
    def __init__(self,input1_dim,input2_dim,out_dim):
        super(Attention, self).__init__()
        self.input1_dim = input1_dim
        self.input2_dim = input2_dim
        self.out_dim = out_dim
        self.attn = nn.Linear(input1_dim,input2_dim)
        self.map = nn.Linear(input2_dim,out_dim)
    def forward(self,key, value):
        
        key = torch.transpose(key,0,1)
        key = self.attn(key) 
        weights = torch.transpose(torch.matmul(key,value),1,2) 
        output = torch.matmul(value,weights).squeeze(2)
        output = self.map(output)
        return output

class DecoderCell(nn.Module):
    def __init__(self, config_dict, slave=False):
        super().__init__()
        input_dim = config_dict['decoder'].get('input_dim', 256)
        hidden_dim = config_dict['decoder'].get('hidden_dim', 256)
        drop_out = config_dict.get('drop_out', 0.2)
        if slave:
            input_dim += hidden_dim
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bias=True, batch_first=True,
                          dropout=drop_out, bidirectional=False)

    def forward(self, decoder_input, hidden):
        '''

        Args:
            decoder_input:      [b, dim]
            hidden:             [b, hidden_dim]
        Returns:

        '''
        output, hidden = self.gru(decoder_input.unsqueeze(1), hidden)
        return output.squeeze(1), hidden


class Decoder(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        encoder_final_out_dim = config_dict['encoder'].get('final_out_dim', 256)
        decoder_hidden_dim = config_dict['encoder'].get('hidden_dim', 256)
        vocabulary_dim = config_dict.get('vocabulary_dim', 1)
        embedding_dim = config_dict.get('embedding_dim', 256)
        input_dim = config_dict['decoder'].get('input_dim', 256)
        hidden_dim = config_dict['decoder'].get('hidden_dim', 256)
        drop_out = config_dict.get('drop_out', 0.2)
        self.W_e2d = nn.Linear(encoder_final_out_dim+config_dict['style_attn']['style_in'], decoder_hidden_dim, bias=True)
        self.word_emb_layer = None
        self.attention_layer = ScaleDotAttention(config_dict)
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bias=True,
                          batch_first=True, bidirectional=False)
        self.projection_layer = nn.Linear(hidden_dim, vocabulary_dim)
        self.mode = 'train'

        self.style_attn = Attention(config_dict['decoder'].get('hidden_dim', 256),config_dict['style_attn']['style_in'],config_dict['style_attn']['style_out'])


    def forward(self, encode_hidden, encode_output, encoder_mask, seq_label, decoder_input, style_emb, max_seq_len=21):
        style_feature = style_emb[:, -1, :]
        encode_hidden = torch.cat([encode_hidden,style_feature],dim=-1)

        hidden = self.W_e2d(encode_hidden).unsqueeze(0)

        if self.mode in ['train', 'eval']:
            decoder_input_emb = self.word_emb_layer(decoder_input)
            decoder_output_arr = []

            for t in range(decoder_input.size()[-1]):
                context, _ = self.attention_layer(hidden[-1], encode_output, encode_output, encoder_mask)
                output, hidden = self.gru(torch.cat([context,decoder_input_emb[:, t]], dim=-1).unsqueeze(dim=1), hidden)
                decoder_output_arr.append(output.squeeze(dim=1))
            decoder_output = self.projection_layer(torch.stack(decoder_output_arr, dim=1))
            if self.mode == 'eval':
                loss = get_nll_loss(decoder_output, seq_label, reduction='none')
                ppl = loss.exp()
                return ppl, loss.mean()
            else:
                loss = get_nll_loss(decoder_output, seq_label, reduction='mean')
                return loss
        elif self.mode == 'infer':
            id_arr = []
            previous_vec = self.word_emb_layer(
                torch.ones(size=[encode_output.size()[0]], dtype=torch.long, device=device) *
                torch.tensor(2,  dtype=torch.long, device=device))
            for t in range(max_seq_len):
                context, _ = self.attention_layer(hidden[-1], encode_output, encode_output, encoder_mask)
                output, hidden = self.gru(torch.cat([context,previous_vec], dim=-1).unsqueeze(dim=1), hidden)
                decode_output = self.projection_layer(output.squeeze(dim=1))
                _, previous_id = decode_output.max(dim=-1, keepdim=False)
                previous_vec = self.word_emb_layer(previous_id)
                id_arr.append(previous_id)
            decoder_id = torch.stack(id_arr, dim=1)
            return decoder_id
        else:
            return None

