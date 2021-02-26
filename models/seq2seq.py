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

class Seq2Seq(nn.Module):

    def __init__(self, opt, encoder, decoder):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.enc_rnn = encoder.enc_rnn
        self.enc_emb_layer = encoder.emb_layer
        self.enc_lin = encoder.enc_lin

        self.max_seq_len = opt["max_seq_len"]
        self.vocab_sz = opt["vocab_sz"]
 

    def forward(self, phrase, sim_phrase, training_mode = True):
        enc_phrase = self.encoder(phrase = phrase.t())
        out = self.decoder(phrase = phrase.t(), enc_phrase = enc_phrase, similar_phrase = sim_phrase.t(), teacher_forcing = False)

        if training_mode:
            #print('similar_phrase_shape', sim_phrase.shape)
            emb_sim_phrase = self.enc_emb_layer(one_hot(sim_phrase.t(),self.vocab_sz))
            #print('emb_sim_phrase_shape', emb_sim_phrase.shape)
            emb_sim_phrase_rnn = self.enc_rnn(emb_sim_phrase)[1]
            #print('emb_sim_phrase_rnn_shape', emb_sim_phrase_rnn.shape)
            enc_sim_phrase = self.enc_lin(emb_sim_phrase_rnn)
            #print('enc_sim_phrase_shape', enc_sim_phrase.shape)
                    
        
            #print('out_shape', out.shape)
            emb_out = self.enc_emb_layer(torch.exp(out))
            #print('emb_out_shape', emb_out.shape)
            emb_out_rnn = self.enc_rnn(emb_out)[1]
            #print('emb_out_rnn_shape', emb_out_rnn.shape)
            enc_out = self.enc_lin(emb_out_rnn)
            #print('enc_out_shape', enc_out.shape)

            enc_out.squeeze_(0)
            enc_sim_phrase.squeeze_(0)
            return out, enc_out, enc_sim_phrase
        else:
            return out
