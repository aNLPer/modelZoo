import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


# 模拟数据
# E 表示 end of sentence
# S 表示 Start of sentence
# P 表示 Padding and it Should be Zero
sentences = [
        # enc_input           dec_input         dec_output
        ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

src_vocab = {'P' : 0, 'ich' : 1, 'mochte' : 2, 'ein' : 3, 'bier' : 4, 'cola' : 5}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P' : 0, 'i' : 1, 'want' : 2, 'a' : 3, 'beer' : 4, 'coke' : 5, 'S' : 6, 'E' : 7, '.' : 8}
tgt_vocab_size = len(tgt_vocab)

idx2word = {i: w for i, w in enumerate(tgt_vocab)}


src_len = 5 # enc_input max sequence length
tgt_len = 6 # dec_input(=dec_output) max sequence length


def MakeData():
   enc_inputs, dec_inputs, dec_outputs = [], [], []
   for i in range(len(sentences)):
       enc_inputs.append([src_vocab[c] for c in sentences[i][0].split()])
       dec_inputs.append([tgt_vocab[c] for c in sentences[i][1].split()])
       dec_outputs.append([tgt_vocab[c] for c in sentences[i][2].split()])

   return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

enc_inputs, dec_inputs, dec_outputs = MakeData()


#
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, item):
        return self.enc_inputs[item], self.dec_inputs[item], self.dec_outputs[item]



class PositionEncoding(nn.Module):
    """
    对输入的序列进行位置嵌入
    """
    def __init__(self):
        super(PositionEncoding, self).__init__()
