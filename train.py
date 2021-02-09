import os
import subprocess
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
#from tensorboardX import SummaryWriter
from tqdm import tqdm

from utilities import train_utils, model_utils
from dataloader import Dataloader
from utilities.train_utils import dump_samples, evaluate_scores, save_model
from models.encoder import Encoder
from models.decoder import Decoder
from models.paraphrase import Paraphrase


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def train():
    # Store parser arguements in a variable
    parser = model_utils.make_parser()
    args = parser.parse_args()

    #load the data
    data = Dataloader(args.input_json, args.input_ques_h5)

    opt = {
        "vocab_sz": data.getVocabSize(),
        "max_seq_len": data.getSeqLength(),
        "emb_hid_dim": args.emb_hid_dim,
        "emb_dim": args.emb_dim,
        "enc_dim": args.enc_dim,
        "enc_dropout": args.enc_dropout,
        "enc_rnn_dim": args.enc_rnn_dim,
        "gen_rnn_dim": args.dec_rnn_dim,
        "gen_dropout": args.dec_dropout,
        "lr": args.learning_rate,
        "epochs": args.n_epoch,
        "layers": 2
    }

    # instanstiate the model
    enc = Encoder(opt)
    dec = Decoder(opt)
    para_model = Paraphrase(opt, enc, dec)

    os.makedirs(os.path.join(GEN_DIR, TIME), exist_ok=True)

    train_loader = Data.DataLoader(
        Data.Subset(data, range(args.train_dataset_len)),
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = Data.DataLoader(
        Data.Subset(data, range(args.val_dataset_len)),
        batch_size = args.batch_size,
        shuffle = True
    )

    #initialize the weights
    para_model.apply(init_weights)
    # optimizer
    optim = optim.Adadelta(para_model.parameters(), lr = opt['lr'])
    para_model.train()

    para_model.to(DEVICE)
    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=data.PAD)

    for epoch in range(opt['epochs']):
        epoch_l1 = 0
        epoch_l2 = 0
        itr = 0
        ph = []
        pph = []
        gpph = []
        para_model.train()




if __name__ == "__main__":

    LOG_DIR = 'logs'
    SAVE_DIR = 'save'
    GEN_DIR = 'samples'
    HOME = './'
    TIME = time.strftime("%Y%m%d_%H%M%S")
    DEVICE = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    train()