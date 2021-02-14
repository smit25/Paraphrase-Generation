import torch
import torch.nn as nn
from random import random

"""
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

"""
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def one_hot(phrase, c):
    return torch.zeros(*phrase.size(), c, device=device).scatter_(-1, phrase.unsqueeze(-1), 1)

class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()

        self.max_seq_len = opt["max_seq_len"]
        self.vocab_sz = opt["vocab_sz"]
        self.layers = 2

        self.dec_emb = nn.Embedding(opt["vocab_sz"], opt["emb_dim"])

        self.dec_rnn = nn.GRU(opt["emb_dim"], opt["dec_rnn_dim"], bidirectional = True)

        self.dec_lin = nn.Sequential(
            nn.Dropout(opt["dec_dropout"]),
            nn.Linear(opt["dec_rnn_dim"], opt["vocab_sz"]),
            nn.LogSoftmax(dim =-1)
        )

    def forward(self, phrase, enc_phrase, similar_phrase = None, teacher_forcing_ratio = 0.5):
        """
        similar_phrase : (if teacher_forcing == True), shape = (max seq length, batch_sz)
        phrase : given phrase , shape = (max sequence length, batch size)
        out : generated paraphrase, shape = (max sequence length, batch size)

        """
        if similar_phrase == None:
            similar_phrase = phrase
        # print('phrase', phrase.shape) # (28, 100)
        # print('similar_phrase', similar_phrase.shape) # (28, 100)
        # print('encoded phrase', enc_phrase.shape) # (1, 100, 512)
        
        # WITH FORCED TEACHER TRAINING
        words = []
        h = None
        emb_sim_phrase_dec = self.dec_emb(similar_phrase)
        #print('emb_sim_phrase', emb_sim_phrase_dec.shape) # (28, 100, 512)
        dec_rnn_inp = torch.cat([enc_phrase, emb_sim_phrase_dec[:-1, :]], dim=0)
        #print('dec_rnn_inp', dec_rnn_inp.shape)
        out_rnn, _ = self.dec_rnn(dec_rnn_inp)
        out = self.dec_lin(out_rnn)

        # WITHOUT FORCED TEACHER TRAINING
        # for __ in range(self.max_seq_len):
        #     word, h = self.dec_rnn(enc_phrase, hx=h)
        #     word = self.dec_lin(word)
        #     words.append(word)
        #     word = torch.multinomial(torch.exp(word[0]), 1)
        #     word = word.t()
        #     enc_phrase = self.dec_emb(word)
        # out = torch.cat(words, dim=0).to(device)
        
        return out
