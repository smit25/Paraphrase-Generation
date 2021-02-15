import torch
import pickle
from utilities.prepro_utils import prepro_input
from modify.antonym import Antonym
from modify.synonym import Synonym

"""
input_array, wtoi, itow = prepro_input(input_sent)
input_len = input_array.size()[0]
input_tensor = torch.from_numpy(input_array.astype(int))

new_tensor = torch.zeros(input_tensor.size()[0]+2, dtype = torch.long)
new_tensor[1: input_len + 1] = input_tensor[0:input_len]
new_tensor[0], new_tensor[input_len + 1] = len(itow)-1, len(itow) -2
"""

working_model = torch.load('para_model.pt') # loading the Seq2Seq model
working_model.eval()

def arr_to_ten(array, wtoi):
  tensor = torch.from_numpy(array.astype(int))
  n = tensor.size()[0]
  new_tensor = torch.zeroes(n+2, dtype = torch.long)
  new_tensor[1:n+1] = tensor[:n]
  new_tensor[0] = wtoi['<SOS>']
  new_tensor[n+1] = wtoi['<EOS>']
  return new_tensor

def decode_seq(itow, seq):
  row, col = seq.size()[0], seq.size()[1]
  output = []
  for i in range(row):
    txt = ''
    for j in range(col):
      index = seq[i, j]
      if int(index.item()) not in itow:
        #print("UNK token ", str(index.item()))
        word = itow[len(itow) - 1]
      else:
        word = itow[int(index.item())]
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

def main():
  input_sent = 'A beautiful castle '

  input_array, w_to_i, i_to_w = prepro_input(input_sent)
  input_tensor = arr_to_ten(input_array, w_to_i)
  input_tensor = torch.unsqueeze(input_tensor, 0) # Add extra dimension

  output = working_model(input_tensor, training_mode = False)
  print(output)

  paraphrase = decode_seq(i_to_w, torch.argmax(output, dim =-1).t())
  print(paraphrase)

  syn = Synonym(paraphrase)
  syn_para = syn.main()
  ant = Antonym(paraphrase)
  ant2 = Antonym(syn_para)
  ant_para, ant2_para = ant.main(), ant2.main()





