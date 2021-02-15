import os
#import subprocess
import time
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utilities import train_utils, model_utils
from dataloader import Dataloader
from utilities.train_utils import dump_samples, evaluate_scores, save_model, joint_emb_loss, decode_sequence
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq

"""
To view the data written by tensorboardX
tensorboard --logdir <path of logs directory>
In my case, pathdir = 'logs/'

"""

train_len = 100000
val_len = 10000
# function to initialize model weights
def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def train():
    # store parser arguements in a variable
    parser = model_utils.make_parser(train_len, val_len)
    args = parser.parse_args()

    #load the data
    data = Dataloader(args.input_json, args.input_ques_h5)
    print('Dataloader successful!')
    logger = SummaryWriter(os.path.join(LOG_DIR, TIME + args.name))

    opt = {
        "vocab_sz": data.getVocabSize(),
        "max_seq_len": 28,
        "emb_hid_dim": args.emb_hid_dim,
        "emb_dim": args.emb_dim,
        "enc_dim": args.enc_dim,
        "enc_dropout": args.enc_dropout,
        "enc_rnn_dim": args.enc_rnn_dim,
        "dec_rnn_dim": args.dec_rnn_dim,
        "dec_dropout": args.dec_dropout,
        "lr": args.learning_rate,
        "epochs": args.n_epoch,
        "layers": 2
    }

    # instanstiate the model
    enc = Encoder(opt)
    dec = Decoder(opt)
    para_model = Seq2Seq(opt, enc, dec)
    print('Models successfully loaded')

    os.makedirs(os.path.join(GEN_DIR, TIME), exist_ok=True)

    #Load training data
    train_loader = Data.DataLoader(
        Data.Subset(data, range(args.train_dataset_len)),
        batch_size=args.batch_size,
        shuffle=True
    )
    #print('----', len(train_loader))

    # Load validation test data
    val_loader = Data.DataLoader(
        Data.Subset(data, range(args.val_dataset_len)),
        batch_size = args.batch_size,
        shuffle = True
    )

    #initialize the weights
    para_model.apply(init_weights)
    # optimizer
    optimizer = optim.Adam(para_model.parameters(), lr = opt['lr'])
    # set to training mode
    para_model.train()

    para_model.to(DEVICE)
    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=data.PAD)
    
    # TRAINING
    print('----------- BEGIN TRAINING -------------')
    for epoch in range(1,opt['epochs']+1):
        epoch_l1 = 0
        epoch_l2 = 0
        itr = 0
        para_model.train()

        for phrase, phrase_len, paraphrase, paraphrase_len, _ in tqdm(
                train_loader, ascii=True, desc="train" + str(epoch)):

            phrase = phrase.to(DEVICE)
            paraphrase = paraphrase.to(DEVICE)

            out, enc_out, enc_sim_phrase = para_model(phrase = phrase,sim_phrase=paraphrase, training_mode= True)
            # print('shape of output tensor', out.shape)
            # print('shape of permuted out', out.permute(1,2,0).shape)
            # print('shape of paraphrase', paraphrase.shape)
            # print('enc_out shape', enc_out.shape)
            # print('enc_sim_phrase', enc_sim_phrase.shape)

            loss_1, loss_2 = cross_entropy_loss(out.permute(1, 2, 0), paraphrase), joint_emb_loss(enc_out, enc_sim_phrase)

            optimizer.zero_grad()
            (loss_1 + loss_2).backward()

            optimizer.step()

            epoch_l1 += loss_1.item()
            epoch_l2 += loss_2.item()

            ph = []
            pph = []
            gpph = []

            ph_solo= decode_sequence(data.index_to_word, phrase)
            print('decode_seq_phrase', ph_solo)
            ph+=ph_solo
            pph_solo = decode_sequence(data.index_to_word, paraphrase)
            print('decode_seq_sim_phrase', pph_solo)
            pph+=pph_solo
            gpph_solo = decode_sequence(data.index_to_word, torch.argmax(out, dim=-1).t())
            print('generated_pph', gpph_solo)
            gpph+=gpph_solo

            itr += 1
            torch.cuda.empty_cache()

        logger.add_scalar("l2_train", epoch_l2 / itr, epoch)
        logger.add_scalar("l1_train", epoch_l1 / itr, epoch)
        # Evaluate bleu, meteor, cider and rouge scores for training set
        score = evaluate_scores(gpph, pph)
        for key in score:
            logger.add_scalar(key + "_train", score[key], epoch)

        dump_samples(ph, pph, gpph, os.path.join(GEN_DIR, TIME,
                                  str(epoch) + "_train.txt"))

        # VALIDATION
        epoch_l1 = 0
        epoch_l2 = 0
        itr = 0
        para_model.eval()

        print('----------- BEGIN VALIDATION -------------')
        for phrase, phrase_len, paraphrase, paraphrase_len, _ in tqdm(
            val_loader, ascii=True, desc="validation" + str(epoch)):

            phrase = phrase.to(DEVICE)
            paraphrase = paraphrase.to(DEVICE)

            # enc_phrase = enc(phrase = phrase.t(), train = True)
            # dec_out = dec(phrase = phrase.t(), enc_phrase = enc_phrase, similar_phrase = paraphrase, teacher_forcing_ratio = 0.6, train = True)

            out, enc_out, enc_sim_phrase = para_model(phrase = phrase,sim_phrase=paraphrase, training_mode=True)

            loss_1, loss_2 = cross_entropy_loss(out.permute(1, 2, 0), paraphrase),joint_emb_loss(enc_out, enc_sim_phrase)

            epoch_l1 += loss_1.item()
            epoch_l2 += loss_2.item()

            ph = []
            pph = []
            gpph = []

            ph_solo_val= decode_sequence(data.index_to_word, phrase)
            ph += ph_solo_val
            pph_solo_val = decode_sequence(data.index_to_word, paraphrase)
            pph += pph_solo_val
            gpph_solo_val = decode_sequence(data.index_to_word, torch.argmax(out, dim=-1).t())
            gpph += gpph_solo_val

            itr += 1
            torch.cuda.empty_cache()
        
        logger.add_scalar("l2_val", epoch_l2 / itr, epoch)
        logger.add_scalar("l1_val", epoch_l1 / itr, epoch)

        # Evaluate bleu, meteor, cider and rouge scores for val set
        score = evaluate_scores(gpph, pph)
        for key in score:
            logger.add_scalar(key + "_val", score[key], epoch)
        dump_samples(ph, pph, gpph,os.path.join(GEN_DIR, TIME, str(epoch) + "_val.txt"))

        # Save model
    save_model(para_model, optimizer, epoch, 'para_model.pt')

    print('Training Done')

if __name__ == "__main__":

    LOG_DIR = 'logs'
    SAVE_DIR = 'save'
    GEN_DIR = 'samples'
    HOME = './'
    TIME = time.strftime("%Y%m%d_%H%M%S")
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train()