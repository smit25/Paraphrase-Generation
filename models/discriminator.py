import torch
import torch.nn as nn
from random import random

import utilities.model_utils as utils

"""
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

"""
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Paraphrase(nn.module):

    def __init__(self, opt, encoder, decoder):
        super(Paraphrase).__init__()

        encoder = self.encoder
        decoder = self.decoder

        self.enc_rnn = encoder.enc_rnn
        self.enc_emb_layer = encoder.enc_emb_layer
        self.enc_lin = encoder.enc_lin
 
        self.max_seq_len = opt["max_seq_len"]
        self.vocab_sz = opt["vocab_sz"]

    def forward(self, phrase, sim_phrase, out):

        enc_sim_phrase = self.enc_lin(
                self.enc_rnn(
                    self.enc_emb_layer(utils.one_hot(sim_phrase,self.vocab_sz)))[1])
        
        enc_out = self.enc_lin(self.enc_rnn(self.enc_emb_layer(torch.exp(out)))[1])

        enc_out.squeeze_(0)
        enc_sim_phrase.squeeze_(0)
        return out, enc_out, enc_sim_phrase
