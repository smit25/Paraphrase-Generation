import torch
import utilities.model_utils as utils
from score_eval.eval import COCOEvalCap
import torch.nn as nn
import os

# Save the model after training
def save_model(model, model_optim, epoch, save_file):
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
def decode_sequence(index_to_word, ppn_dict_og, ppn_dict_d, seq, idx, str):
    types1 = [type(k) for k in index_to_word.keys()]
    #print(types1[0])
    # ppn_list = []
    # if str == 'para':
    #     ppn_list = [ppn_dict_og[idx] if len(ppn_dict_og[idx]) > len(ppn_dict_d[idx]) else ppn_dict_d[idx]]
    # elif str == 'og':
    #     ppn_list = ppn_dict_og[idx]
    # else:
    #     ppn_list = ppn_dict_d[idx]

    row, col = seq.size()[0], seq.size()[1]
    output = []
    print('seq', seq)
    for i in range(row):
        txt = ''
        SOS_flag = False
        ppn_count = 0
        for j in range(col):
            index = seq[i, j]
            #print('index', index)
            #print('index.item()', type(index.item()))
            if index_to_word.get(str(index.item())) == None:
                #print('Smit', len(index_to_word) -1)
                word = index_to_word[str(len(index_to_word)-1)] # UNK Token
            else:
                word = index_to_word[str(index.item())]
            if word == 'UNK' and ppn_count < len(ppn_list):
                word = ppn_list[ppn_count]
                ppn_count+=1
            if word == '<EOS>':
                txt = txt + ' ' + word
                break
            if word == '<SOS>' and not SOS_flag:
                txt += '<SOS>'
                SOS_flag = True
                continue
            if j > 0 and word != '<SOS>':
                txt = txt + ' '
            if not SOS_flag or word != '<SOS>':
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
    class score:
        def __init__(self, sents):
            self.sents = sents
            self.imgToAnns = [[{'caption': sents[i]}] for i in range(len(sents))]

        def getImgIds(self):
            return [i for i in range(len(self.sents))]

    return score(real_sents), score(pred_sents)


def evaluate_scores(s1, s2):

    '''
    calculates scores and return the dict with score_name and value
    '''
    score, scoreRes = getObjsForScores(s1, s2)
    evalObj = COCOEvalCap(score, scoreRes)
    evalObj.evaluate()

    return evalObj.eval
