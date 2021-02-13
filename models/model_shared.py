import torch
import torch.nn as nn
from random import random

import utilities.utils as utils

"""
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

"""
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Paraphrase(nn.Module):

    def __init__(self, opt):
        super(Paraphrase).__init__()

        # ENCODER
        self.emb_layer = nn.Sequential(
            nn.Linear(opt["vocab_sz"], opt["emb_hid_dim"]),
            nn.Threshold(0.000001, 0),
            nn.Dropout(opt["enc_dropout"]),
            nn.Linear(opt["emb_hid_dim"], opt["emb_dim"]),
            nn.Threshold(0.000001, 0))
        
        self.enc_rnn = nn.GRU(opt["emb_dim"], opt["enc_rnn_dim"])

        self.enc_lin = nn.Sequential(
            nn.Dropout(opt["enc_dropout"]),
            nn.Linear(opt["enc_rnn_dim"], opt["enc_dim"]))

        # DECODER
        self.dec_emb = nn.Embedding(opt["vocab_sz"], opt["emd_dim"])
        self.dec_rnn = nn.GRU(opt["emb_dim"], opt["dec_rnn_dim"])
        self.dec_lin = nn.Sequential(
            nn.Dropout(opt["dec_dropout"]),
            nn.Linear(opt["dec_rnn_dim"], opt["vocab_sz"]),
            nn.LogSoftmax(dim =-1)
        )

        # DISCRIMINATOR AND ENCODER SHARE WEIGHTS

        self.max_seq_len = opt["max_seq_len"]
        self.vocab_sz = opt["vocab_sz"]

    def forward(self, phrase, similar_phrase = None, teacher_forcing_ratio = 0.5):
        """
        inputs :-
        phrase : given phrase , shape = (max sequence length, batch size)
        sim_phrase : (if teacher_forcing == True), shape = (max seq length, batch_sz)
        teacher_forcing : if true teacher forcing is used to train the module

        outputs :-
        out : generated paraphrase, shape = (max sequence length, batch size)
        enc_out : encoded generated paraphrase, shape=(batch size, enc_dim)
        enc_sim_phrase : encoded sim_phrase, shape=(batch size, enc_dim)

        """
        if similar_phrase == None:
            similar_phrase = phrase

        # encode input phrase
        enc_phrase = self.enc_lin(
            self.enc_rnn(
                self.emb_layer(utils.one_hot(phrase, self.vocab_sz)))[1])

        # Using real target outputs as each next input, instead of using the decoder’s guess as the next input. 
        # Manually concatenate the final states for both directions  
        if random.random() < teacher_forcing_ratio:
            emb_sim_phrase_dec = self.dec_emb(similar_phrase)
            out_rnn, _ = self.dec_rnn(
                torch.cat([enc_phrase, emb_sim_phrase_dec[:-1, :]], dim=0))
            out = self.dec_lin(out_rnn)

        else:
            words = []
            h = None
            for __ in range(self.max_seq_len):
                word, h = self.dec_rnn(enc_phrase, hx=h)
                word = self.dec_lin(word)
                words.append(word)
                word = torch.multinomial(torch.exp(word[0]), 1)
                word = word.t()
                enc_phrase = self.dec_emb(word)
            out = torch.cat(words, dim=0)
        
        # encode similar phrase and generated output to calculate pair-wise discriminator loss
        enc_sim_phrase = self.enc_lin(
                self.enc_rnn(
                    self.enc_emb_layer(utils.one_hot(similar_phrase,self.vocab_sz)))[1])
        
        enc_out = self.enc_lin(self.enc_rnn(self.enc_emb_layer(torch.exp(out)))[1])

        enc_out.squeeze_(0)
        enc_sim_phrase.squeeze_(0)
        return out, enc_out, enc_sim_phrase

