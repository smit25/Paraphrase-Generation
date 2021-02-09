  
import torch
import utilities.model_utils as utils
#import utilities.net_utils as net_utils
import torch.nn as nn
import os

def save_model(epoch, model, model_optim, save_file):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'model_opt': model_optim.state_dict()
    }

    torch.save(checkpoint, save_file)


def dump_samples(ph, pph, gpph, file_name):
    file = open(file_name, "w")

    for r, s, t in zip(ph, pph, gpph):
        file.write("ph : " + r + "\npph : " + s + "\ngpph : " + t + '\n\n')
    file.close()

def decode_sequence(ix_to_word, seq):
    N, D = seq.size()[0], seq.size()[1]
    output = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if int(ix.item()) not in ix_to_word:
                print("UNK token ", str(ix.item()))
                word = ix_to_word[len(ix_to_word) - 1]
            else:
                word = ix_to_word[int(ix.item())]
            if word == '<EOS>':
                txt = txt + ' ' + word
                break
            if word == '<SOS>':
                txt += '<SOS>'
                continue
            if j > 0:
                txt = txt + ' '
            txt += word
        output.append(txt)
    return output


def joint_emb_loss(emb1, emb2):
    batch_size = emb1.size()[0]
    return torch.sum(
        torch.clamp(
            torch.mm(emb1, emb2.t()) - torch.sum(emb1 * emb2, dim=-1) + 1,min=0.0)
    ) / (batch_size * batch_size)


def prob_to_pred(prob):
    return torch.multinomial(torch.exp(prob.view(-1, prob.size(-1))), 1).view(prob.size(0), prob.size(1))


def getObjsForScores(real_sents, pred_sents):
    class coco:
        def __init__(self, sents):
            self.sents = sents
            self.imgToAnns = [[{'caption': sents[i]}] for i in range(len(sents))]

        def getImgIds(self):
            return [i for i in range(len(self.sents))]

    return coco(real_sents), coco(pred_sents)


def evaluate_scores(s1, s2):

    '''
    calculates scores and return the dict with score_name and value
    '''
    coco, cocoRes = getObjsForScores(s1, s2)

    evalObj = COCOEvalCap(coco, cocoRes)

    evalObj.evaluate()

    return evalObj.eval
