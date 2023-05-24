import torch
import torch.nn as nn
from torch.distributions import Categorical
import math
from tqdm import tqdm
import numpy as np
from collections import deque
from torch.nn.functional import pad
import math
import re

from data.tokens import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, TOKENS_DICT, token_to_id, id_to_token, TOKENS_GROUP_THREE



# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, learn_pos, max_len):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
        self.learn_pos = learn_pos
        # max_len+2: for eos / bos token
        self.positional_embedding = nn.Parameter(torch.randn([1, max_len+2, emb_size]))
        
    def forward(self, tokens):
        x = self.embedding(tokens.long()) * math.sqrt(self.emb_size)
        x_batch_size = x.shape[0]
        x_seq_len = x.shape[1]
        if self.learn_pos:
            pe = self.positional_embedding[:,:x_seq_len]
            pe_stack = torch.tile(pe, (x_batch_size, 1, 1))
            return x+pe_stack
        return x

class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: shape [batch_size, seq_len, emb_size (pad_vocab_size)]
        x = x + self.pe[:x.size(1), :].transpose(0,1)
        return x

class TreePositionalEncoding(nn.Module):
    def __init__(self, d_model, token2id, pos_type, max_len, k=2):
        super().__init__()
        self.d_model = d_model
        self.pos_type = pos_type
        l = range(k**2+1)
        position_dict = {str(key): np.zeros(len(l)-1, dtype=int) for key in l}
        for key, value in position_dict.items():
            position_dict[key][int(key)-1] = 1
        position_dict['0'][-1] = 0
        position_dict = {key: tuple(value) for key, value in position_dict.items()}
        
        self.k = k
        self.k_square = k**2
        # self.pos_dict = {'0': (0,0,0,0), '1': (1,0,0,0), '2': (0,1,0,0), '3': (0,0,1,0), '4': (0,0,0,1)}
        self.pos_dict = position_dict
        self.token2id = token2id
        self.bos = self.token2id[BOS_TOKEN]
        self.eos = self.token2id[EOS_TOKEN]
        self.pad = self.token2id[PAD_TOKEN]
        if 'group' in self.pos_type:
            self.max_pe_length = int(math.log(max_len, self.k_square)+5)
        else:
            self.max_pe_length = int(math.log(max_len, self.k_square)+5)*(k**2)
        self.positional_embedding = nn.Linear(self.max_pe_length, self.d_model)
        self.padder = torch.nn.ReplicationPad2d((0,self.d_model-self.max_pe_length,0,0))
        is_group_dict = {'emb': False, 'group-emb': True, 'pad': False, 'group-pad': True}
        self.is_group = is_group_dict[self.pos_type]
        
    def filter_row_string(self, row_string):
        # filter bos to eos
        l = row_string.tolist()
        try:
            return l[1:l.index(self.eos)]
        except ValueError:
            return l[1:]
    
    def get_pe(self, row_string):
        l = self.filter_row_string(row_string)
        if len(l) == 0:
            return torch.zeros((1,1))
        elif len(l) > 3:
            string = l[0:self.k_square]
            pos_list = list(range(1,self.k_square+1))
        else:
            string = l[0:len(l)]
            pos_list = list(range(1,len(l)+1))
            
        str_pos_queue = deque([(s, p) for s, p in zip(string, pos_list)])
        for i in np.arange(self.k_square,len(l),self.k_square):
            cur_string = l[i:i+self.k_square]
            cur_parent, cur_parent_pos = str_pos_queue.popleft()
            # if value is 0, it cannot be parent node -> skip
            while((cur_parent == self.token2id['0']) and (len(str_pos_queue) > 0)):
                cur_parent, cur_parent_pos = str_pos_queue.popleft()
            # i: order of the child node in the same parent
            cur_pos = [cur_parent_pos*10+i for i in range(1,1+len(cur_string))]
            # pos_list: final position of each node
            pos_list.extend(cur_pos)
            str_pos_queue.extend([(s, c) for s, c in zip(cur_string, cur_pos)])
        # map position vector to each position
        reverse_pos_list = [str(pos)[::-1] for pos in pos_list]
        tensor_pos_list = [self.map_pos_to_tensor(pos) for pos in reverse_pos_list]
        max_size = len(tensor_pos_list[-1])
        final_pos_list = [pad(pos, (0,max_size-len(pos))) for pos in tensor_pos_list]
        
        # return shape: sequence len * pe size
        return torch.stack(final_pos_list)
    
    def pe(self, pe_tensor, emb_size):
        return self.positional_embedding(pe_tensor) * math.sqrt(emb_size)
    
    def map_pos_to_tensor(self, pos):
        result = []
        for char in pos:
            result.extend(self.pos_dict[char])
            
        return torch.tensor(result)
    
    def map_pos_to_tensor_group(self, pos):
        return torch.tensor([eval(p) for p in pos])
    
    def finalize_pe(self, pos_list, is_group):
        if is_group:
            int_pos_list = [list(str(pos)) for pos in pos_list]
            tensor_pos_list = [self.map_pos_to_tensor_group(pos) for pos in int_pos_list]
        else:
            tensor_pos_list = [self.map_pos_to_tensor(str(pos)) for pos in pos_list]
        max_size = len(tensor_pos_list[-1])
        final_pos_list = [pad(pos, (0,max_size-len(pos))) for pos in tensor_pos_list]
        return torch.stack(final_pos_list)
    
    def forward(self, x, org_x):
        # x shape: batch size * sequence len * emb size
        if x.size(1) == 1:
            return x
        # pe shape: sequence len * pe size
        pos_lists = [self.get_pe(x_string) for x_string in org_x]
        pe_list = [self.finalize_pe(pos_list, is_group=self.is_group) for pos_list in pos_lists]
        max_str_length = x.shape[1]
        pe_tensor = torch.stack([pad(pe, (0,self.max_pe_length-pe.shape[1],1,max_str_length-pe.shape[0]-1)) 
                                for pe in pe_list]).to(x.device).float()
        if 'pad' in self.pos_type:
            pe = self.padder(pe_tensor).to(x.device)
        elif 'emb' in self.pos_type:
            pe = self.pe(pe_tensor, self.d_model)
        
        x += pe
        return x


class GroupTreePositionalEncoding(TreePositionalEncoding):
    def __init__(self, d_model, token2id, id2token, pos_type, max_len, k):
        super().__init__(d_model, token2id, pos_type, max_len, k)
        self.id2token = id2token
    
    def map_string_to_sum(self, raw_string):
        # group position (1,1,0,1,1,1 parent -> 11, 12, 14, 15, 16)
        string = self.id2token[raw_string]
        result = [char.start()+1 for char in re.finditer(r'(?!0).', string)]
        result.append(0)
        return result
    
    def get_pe(self, row_string):
        l = self.filter_row_string(row_string)
        if (len(l) == 0) or (self.pad in l) or (self.bos in l):
            return [0]
        else:
            str_queue = deque(self.map_string_to_sum(l[0]))
            pos_list = [1]
        
        pos_queue = deque(pos_list)
        cur_parent_pos = pos_queue.popleft()
        
        for i in range(1,len(l)):
            tree_index = str_queue.popleft()
            cur_string = l[i]
            if tree_index == 0:
                if len(str_queue) == 0:
                    break
                tree_index = str_queue.popleft()
                cur_parent_pos = pos_queue.popleft()
            
            cur_pos = cur_parent_pos*10+tree_index
            pos_list.append(cur_pos)
            pos_queue.append(cur_pos)
            str_queue.extend(self.map_string_to_sum(cur_string))
        
        return pos_list
    

class TransGenerator(nn.Module):
    '''
    without tree information (only string)
    '''
    
    def __init__(
        self, num_layers, emb_size, nhead, dim_feedforward, 
        input_dropout, dropout, max_len, string_type, tree_pos, pos_type, learn_pos, abs_pos, k
    ):
        super(TransGenerator, self).__init__()
        self.nhead = nhead
        self.tokens = TOKENS_DICT[string_type]
        self.ID2TOKEN = id_to_token(self.tokens)
        self.string_type = string_type
        self.k = k
        self.TOKEN2ID = token_to_id(self.string_type, self.k)
        self.tree_pos = tree_pos
        self.pos_type = pos_type
        self.learn_pos = learn_pos
        self.abs_pos = abs_pos
        self.max_len = max_len
        
        
        if self.abs_pos:
            self.positional_encoding = AbsolutePositionalEncoding(emb_size)
        
        if string_type in ['group', 'bfs-deg-group', 'qm9', 'zinc', 'group-red', 'qm9-red', 'zinc-red', 'group-red-3']:
            if self.tree_pos:
                self.positional_encoding = GroupTreePositionalEncoding(emb_size, self.TOKEN2ID, self.ID2TOKEN, self.pos_type, self.max_len, self.k)
        else:
            if self.tree_pos:
                self.positional_encoding = TreePositionalEncoding(emb_size, self.TOKEN2ID, self.pos_type, self.max_len, self.k)
        #
        self.token_embedding_layer = TokenEmbedding(len(self.tokens), emb_size, self.learn_pos, self.max_len)
        self.input_dropout = nn.Dropout(input_dropout)
        
        #
        self.distance_embedding_layer = nn.Embedding(max_len + 1, nhead)

        #
        encoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu")
        encoder_norm = nn.LayerNorm(emb_size)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        #
        self.generator = nn.Linear(emb_size, len(self.tokens))
        # self.node_type_generator = nn.Linear(emb_size, len())

    def forward(self, sequences):
        
        batch_size = sequences.size(0)
        sequence_len = sequences.size(1)
        TOKEN2ID = token_to_id(self.string_type, self.k)
        #
        out = self.token_embedding_layer(sequences)
        if self.tree_pos:
            out = self.positional_encoding(out, sequences)
        if self.abs_pos:
            out = self.positional_encoding(out)
        out = self.input_dropout(out)


        if self.tree_pos or self.abs_pos:
            mask = torch.zeros(batch_size, sequence_len, sequence_len, self.nhead, device=out.device)
        else:
            # relational positional encoding
            distance_squares = torch.abs(torch.arange(sequence_len).unsqueeze(0) - torch.arange(sequence_len).unsqueeze(1))
            distance_squares[distance_squares > self.max_len] = self.max_len
            distance_squares = distance_squares.unsqueeze(0).repeat(batch_size, 1, 1)
            distance_squares = distance_squares.to(out.device)
            mask = self.distance_embedding_layer(distance_squares)
            
        mask = mask.permute(0, 3, 1, 2)

        #
        bool_mask = (torch.triu(torch.ones((sequence_len, sequence_len))) == 1).transpose(0, 1)
        bool_mask = bool_mask.view(1, 1, sequence_len, sequence_len).repeat(batch_size, self.nhead, 1, 1).to(out.device)
        mask = mask.masked_fill(bool_mask == 0, float("-inf"))
        mask = mask.reshape(-1, sequence_len, sequence_len)

        #
        
        key_padding_mask = sequences == TOKEN2ID[PAD_TOKEN]

        out = out.transpose(0, 1)
        out = self.transformer(out, mask, key_padding_mask)
        out = out.transpose(0, 1)

        #
        logits = self.generator(out)

        return logits

    def decode(self, num_samples, max_len, device):
        TOKEN2ID = token_to_id(self.string_type, self.k)
        sequences = torch.LongTensor([[TOKEN2ID[BOS_TOKEN]] for _ in range(num_samples)]).to(device)
        ended = torch.tensor([False for _ in range(num_samples)], dtype=torch.bool).to(device)
        for _ in tqdm(range(max_len), "generation"):
            if ended.all():
                break
            logits = self(sequences)
            preds = Categorical(logits=logits[:, -1]).sample()
            preds[ended] = TOKEN2ID[PAD_TOKEN]
            sequences = torch.cat([sequences, preds.unsqueeze(1)], dim=1)

            ended = torch.logical_or(ended, preds == TOKEN2ID[EOS_TOKEN])

        return sequences