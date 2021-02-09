import torch
import torch.nn as nn

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

class Encoder(nn.module):

    def __init__(self, opt):
        super(Encoder).__init__()

        self.max_seq_len = opt["max_seq_len"]
        self.vocab_sz = opt["vocab_sz"]

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

    def forward(self, phrase):
        """
        phrase : given phrase , shape = (max sequence length, batch size)

        """
        emb = self.emb_layer(utils.one_hot(phrase, self.vocab_sz))
        out_rnn = self.enc_rnn(emb)[1]
        print('OUT RNN: ', out_rnn)
        enc_phrase = self.enc_lin(out_rnn)

        return enc_phrase
