import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRANSFORMERS_CACHE='/gds/hryang/projdata11/hryang/transformers-cache'
class StyleExtractor(nn.Module):
    def __init__(self,config_dict):
        super().__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased",output_hidden_states=True,cache_dir=TRANSFORMERS_CACHE)

    def forward(self,input):
        outputs = self.model(input)
        hidden_states = torch.stack(outputs[2],dim=1)
        first_hidden_states = hidden_states[:,:,0,:]
        #print(first_hidden_states.shape)[64,13,768]
        #print(outputs[0][:,0,:])
        #print(first_hidden_states[:,-1,:])
        return first_hidden_states

