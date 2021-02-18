import torch
import pickle
from utilities.prepro_utils import prepro_input
from models.seq2seq import Seq2Seq
from modify.antonym import Antonym
from modify.synonym import Synonym
from modify.tenses import Tense

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
working_model = torch.load('para_model.pt', map_location = device) # loading the Seq2Seq model
working_model.eval()

"""
Steps to convert a sentence to tensor:
- prerocess input 
  - tokenize
  - tag the words
  - make word to index map
  - map every word to its corresponding index and make an array
- make a torch tensor from the array (arr_to_ten)
- unsqueeze to add extra dimension
"""

def arr_to_ten(array, wtoi, sent_len, c=1):
  tensor = torch.from_numpy(array.astype(int))
  n = tensor.size()[0]
  new_tensor = torch.zeros(n+2, dtype = torch.long) + wtoi['<PAD>']
  if c==1:
    new_tensor[1:sent_len+1] = tensor[:sent_len]
    new_tensor[0] = wtoi['<SOS>']
    new_tensor[sent_len+1] = wtoi['<EOS>']
  elif c==2:
    b = False
    for i in range(n):
      if tensor[i] == 0 and not b:
        new_tensor[i+1] = wtoi['<EOS>']
        b = True
      if b:
        new_tensor[i+1] = wtoi['<PAD>']
      if not b:
        new_tensor[i+1] = tensor[i]
    new_tensor[0] = wtoi['<SOS>']
  return new_tensor


def decode_seq(itow, seq, ppn_list):
    # types = [type(k) for k in itow.keys()]
    # print(types[1])
    # print(types[28757])
    row, col = seq.size()[0], seq.size()[1]
    output = []
    print('seq', seq)
    txt = ''
    SOS_flag = False
    ppn_count = 0
    print('itow_len', len(itow))
    for j in range(col):
        index = seq[0][j]
        if int(index.item()) not in itow:
                #print('Smit', len(index_to_word) -1)
            word = itow[len(itow)-1] # UNK Token
            print('Smit')
        else:
            word = itow[index.item()]
        print('word: ', word)
        if word == 'UNK' and ppn_count < len(ppn_list):
            word = ppn_list[ppn_count]
            ppn_count+=1
        if word == '<EOS>':
            txt += ' '
            txt += word
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


def main():
  input_sent = 'It is not beautiful'

  input_array, w_to_i, i_to_w, ppn_list, sent_len = prepro_input(input_sent)

  input_tensor = arr_to_ten(input_array, w_to_i, sent_len)
  input_tensor = torch.unsqueeze(input_tensor, 0) # Add extra dimension
  dummy_tensor = torch.zeros(input_tensor.size(), dtype = torch.long)
  print('input_sent', decode_seq(i_to_w, input_tensor, []))
  print('input_tensor', input_tensor)

  output = working_model(input_tensor, input_tensor, training_mode = False)
  #print(output)

  paraphrase = decode_seq(i_to_w, torch.argmax(output, dim =-1).t(), ppn_list)
  print(paraphrase)

  tense = Tense(input_sent, paraphrase)
  tense_rect_out = tense.main()
  syn = Synonym(tense_rect_out)
  syn_out = syn.main()
  print('syn_para', syn_out)
  ant = Antonym(input_sent)
  ant2 = Antonym(syn_out)
  ant_para, ant2_para = ant.main(), ant2.main() 
  # print('ant_para', ant_para)
  # print('ant2_para', ant2_para)
  

if __name__ == '__main__':
  main()





