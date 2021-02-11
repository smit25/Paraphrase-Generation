import torch
import pickle
from utilities.prepro_utils import prepro_input

"""
input_array, wtoi, itow = prepro_input(input_sent)
input_len = input_array.size()[0]
input_tensor = torch.from_numpy(input_array.astype(int))

new_tensor = torch.zeros(input_tensor.size()[0]+2, dtype = torch.long)
new_tensor[1: input_len + 1] = input_tensor[0:input_len]
new_tensor[0], new_tensor[input_len + 1] = len(itow)-1, len(itow) -2
"""

gen_enc = torch.load('') # loading the encoder model
gen_dec = torch.load('') # loading the encoder model
gen_enc.eval()
gen_dec.eval()

input_sent = ''


