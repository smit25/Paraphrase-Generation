import torch
import torch.nn as nn
from random import random

import utilities.utils as utils

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Paraphrase(nn.module):

    def __init__(self, opt):
        super(Paraphrase).__init__()

        self.layers = 2
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

        # DISCRIMINATOR
        self.dis_emb_layer = nn.Sequential(
            nn.Linear(opt["vocab_sz"], opt["emb_hid_dim"]),
            nn.Threshold(0.000001, 0),
            nn.Linear(opt["emb_hid_dim"], opt["emb_dim"]),
            nn.Threshold(0.000001, 0),
        )
        self.dis_rnn = nn.LSTM(opt["emb_dim"], opt["enc_rnn_dim"])
        self.dis_lin = nn.Sequential(
            nn.Dropout(opt["enc_dropout"]),
            nn.Linear(opt["enc_rnn_dim"], opt["enc_dim"]))

        self.max_seq_len = opt["max_seq_len"]
        self.vocab_sz = opt["vocab_sz"]

    def forward(self, phrase, similar_phrase = None, teacher_forcing_ratio = 0.5):
        """
        inputs :-
        phrase : given phrase , shape = (max sequence length, batch size)
        sim_phrase : (if teacher_forcing == True), shape = (max seq length, batch_sz)

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

        hid = None
        words = []
        for __ in range(self.max_seq_len):
        # Using real target outputs as each next input, instead of using the decoder’s guess as the next input.   
            if teacher_forcing_ratio > random.random():
                emb_sim_phrase_dec = self.dec_emb(similar_phrase)
                word, hid = self.dec_rnn(enc_phrase, emb_sim_phrase_dec[:-1, :], hx = hid) 
                word = self.dec_lin(word)
                words.append(word)

            else:
                word, hid = self.dec_rnn(enc_phrase, hx=hid)
                word = self.dec_lin(word)
                words.append(word)

        out = torch.cat(words, dim=0)
        
        # encode similar phrase and generated output to calculate pair-wise discriminator loss
        enc_sim_phrase = self.dis_lin(
                self.dis_rnn(
                    self.dis_emb_layer(utils.one_hot(similar_phrase,self.vocab_sz)))[1])
        
        enc_out = self.dis_lin(self.dis_rnn(self.dis_emb_layer(torch.exp(out)))[1])

        enc_out.squeeze_(0)
        enc_sim_phrase.squeeze_(0)
        return out, enc_out, enc_sim_phrase

