import torch
import utilities.model_utils as utils
from score_eval.eval import COCOEvalCap
import torch.nn as nn
import os

# Save the model after training
def save_model(epoch, model, model_optim, save_file):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'model_opt': model_optim.state_dict()
    }

    torch.save(checkpoint, save_file)

# save the examples 
def dump_samples(ph, pph, gpph, file_name):
    file = open(file_name, "w")

    for r, s, t in zip(ph, pph, gpph):
        file.write("ph : " + r + "\npph : " + s + "\ngpph : " + t + '\n\n')
    file.close()

# Convert the embeddings received form the decoder into a sentence
def decode_sequence(index_to_word, seq):
    row, col = seq.size()[0], seq.size()[1]
    output = []
    for i in range(row):
        txt = ''
        for j in range(col):
            index = seq[i, j]
            if int(index.item()) not in index_to_word:
                #print("UNK token ", str(index.item()))
                word = index_to_word[len(index_to_word) - 1]
            else:
                word = index_to_word[int(index.item())]
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

# Global loss at the discriminator
def joint_emb_loss(emb1, emb2):
    batch_size = emb1.size()[0]
    return torch.sum(
        torch.clamp(
            torch.mm(emb1, emb2.t()) - torch.sum(emb1 * emb2, dim=-1) + 1,min=0.0)
    ) / (batch_size * batch_size)


def prob_to_pred(prob):
    return torch.multinomial(torch.exp(prob.view(-1, prob.size(-1))), 1).view(prob.size(0), prob.size(1))

# Evaluation function
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
