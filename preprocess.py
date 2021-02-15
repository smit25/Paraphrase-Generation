"""
Preprocessing a raw json dataset into json/h5py files.
Experimental protocols mentioned in (Gupta et al., 2017) for the Quora dataset are followed
This script will generate:
- quora_data_prepro.h5 (which will contain the questions and its duplicated encoded into a numpy array)
- quora_data_prepro.json (which will contain the index to word and the word to index dictionary)

"""
import copy
from random import shuffle, seed
import sys
import os.path
import argparse
import glob
import numpy as np
import scipy.io
import pdb
import string
import h5py
from nltk.tokenize import word_tokenize
import json
import re
import math
import pickle
from utilities.prepro_utils import prepro_parser
#from scipy.misc import imread, imresize

maxlen = 26
batch_size = 150
# function for Tokenizing the questions
def tokenize(sentence, params):
    if params['token_method'] == 'nltk':
        return(word_tokenize(sentence))
    else:
        return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n']

# Preprocess(tokenize) all the questions and duplicates
def prepro_question(imgs, params, c=0):
    for i,img in enumerate(imgs):
        # c represents the original or duplicate question in every row
        if c==0:
            s = img['question']
            txt = tokenize(s, params)
            txt = txt[:len(txt)-1]
            img['processed_tokens'] = txt
        elif c==1:
            s = img['question'+str(c)]
            txt = tokenize(s, params)
            txt = txt[:len(txt)-1]
            img['processed_tokens_duplicate'] = txt

        if i < 10: print ('Tokens: ', txt)
        if i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(imgs), i*100.0/len(imgs)) )
            sys.stdout.flush()   
    return imgs

# Replacing less frequent words with 'UNK'
# Build vocabulary for question and duplicates.
def build_vocab_question(imgs, params):
 
    count_thr = params['word_count_threshold']
    counts = {} # word vs its frequency
    
    #count words for every original question in every row
    for img in imgs:
        for w in img['processed_tokens']:
            counts[w] = counts.get(w, 0) + 1

    cw = sorted([(count,w) for w,count in counts.items()], reverse=True) # sort according to frequency in descending order
    print ('Top words and their counts:', '\n'.join(map(str,cw[:20])))

    # print obtained results
    total_words = sum(counts.values())
    print ('total words:', total_words)

    # segregte words on the basis of their frequency relative to the threshold
    low_words = [w for w,n in counts.items() if n <= count_thr]
    vocab = [w for w,n in counts.items() if n > count_thr]   
    low_count = sum(counts[w] for w in low_words)

    print ('number of bad words: %d/%d = %.2f%%' % (len(low_words), len(counts), len(low_words)*100.0/len(counts)))
    print ('number of words in vocab would be %d' % (len(vocab), ))
    print ('number of UNKs: %d/%d = %.2f%%' % (low_count, total_words, low_count*100.0/total_words))

    # Words with less frequency mapped to 'UNK'
    print ('inserting the special UNK token')
    vocab.append('UNK')
    print('Printing vocab', vocab)
    print('\n')
    print(len(vocab))

  
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
        img['final_question'] = question
        txt_d = img['processed_tokens_duplicate']
        duplicate = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt_d]
        img['final_duplicate'] = duplicate
    
    return imgs, vocab

# Applying vocab on validation and test sets for question and duplicate.
def use_vocab_question(imgs, wtoi):  
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if wtoi.get(w,len(wtoi)+1) != (len(wtoi)+1) else 'UNK' for w in txt]
        img['final_question'] = question
        txt_c = img['processed_tokens_duplicate']
        duplicate = [w if w in wtoi else 'UNK' for w in txt_c]
        img['final_duplicate'] = duplicate 

    return imgs

# Embedding the questions into one-hot vector using Bag of Words Model (word2vec)
def encode_question(imgs, params, wtoi):
    max_length = params['max_length'] # of a question
    N = len(imgs)

    # For storing encoded question words
    question_arrays = np.zeros((N, max_length), dtype='uint32')
    question_length = np.zeros(N, dtype='uint32')
    question_id = np.zeros(N, dtype='uint32')
    question_counter = 0
    
    # For storing encoded duplicate words
    duplicate_arrays = np.zeros((N, max_length), dtype='uint32') 
    duplicate_length = np.zeros(N, dtype='uint32')
       
    for i,img in enumerate(imgs):
        question_id[question_counter] = img['id'] #unique_id
        question_length[question_counter] = min(max_length, len(img['final_question'])) # record the length of this question sequence
        duplicate_length[question_counter] = min(max_length, len(img['final_duplicate'])) # record the length of this duplicate sequence        
        question_counter += 1
        for k,w in enumerate(img['final_question']):
            if k < max_length:
                question_arrays[i,k] = wtoi[w]
        for k,w in enumerate(img['final_duplicate']):
            if k < max_length:
                duplicate_arrays[i,k] = wtoi[w]            
  
    return question_arrays, question_length, question_id, duplicate_arrays, duplicate_length

# Calling the functions
def main(params):
    imgs_train5 = json.load(open(params['input_train_json5'], 'r'))
    imgs_test5 = json.load(open(params['input_test_json5'], 'r'))
    
    seed(42)

    # tokenization and preprocessing training question
    imgs_train5 = prepro_question(imgs_train5, params)
    # tokenization and preprocessing test question
    imgs_test5 = prepro_question(imgs_test5, params)

    # tokenization and preprocessing training paraphrase question
    imgs_train5 = prepro_question(imgs_train5, params,1)
    # tokenization and preprocessing test paraphrase question
    imgs_test5 = prepro_question(imgs_test5, params,1)

    # create the vocab for question
    imgs_train5, vocab = build_vocab_question(imgs_train5, params)
    with open('vocab.txt', 'wb') as vocab_save:
        pickle.dump(vocab, vocab_save)
        vocab_save.close()
    
    itow = {i+1:w for i,w in enumerate(vocab)} # 1-indexed vocab translation table
    print('itow', len(itow))
    wtoi = {w:i+1 for i,w in enumerate(vocab)} # Bag of words model
    print('wtoi', len(wtoi))

    ques_train5, ques_length_train5, question_id_train5 , dup_train5, dup_length_train5 = encode_question(imgs_train5, params, wtoi)
    print('ques train shape: ', ques_train5.shape)
    
    imgs_test5 = use_vocab_question(imgs_test5, wtoi)
    
    ques_test5, ques_length_test5, question_id_test5 , dup_test5, dup_length_test5 = encode_question(imgs_test5, params, wtoi)
    
    # H5 FILE PREPRO
    # Creating and writing on the h5 file
    N = len(imgs_train5)
    f = h5py.File(params['output_h5'], "w")

    ## for train data
    f.create_dataset("ques_train", dtype='uint32', data=ques_train5)
    f.create_dataset("ques_length_train", dtype='uint32', data=ques_length_train5)
    f.create_dataset("ques_dup_id_train", dtype='uint32', data=question_id_train5)# ques_dup_id
    f.create_dataset("ques1_train", dtype='uint32', data=dup_train5)
    f.create_dataset("ques1_length_train", dtype='uint32', data=dup_length_train5)

    ## for test data
    f.create_dataset("ques_test", dtype='uint32', data=ques_test5)
    f.create_dataset("ques_length_test", dtype='uint32', data=ques_length_test5)
    f.create_dataset("ques_dup_id_test", dtype='uint32', data=question_id_test5)
    f.create_dataset("ques1_test", dtype='uint32', data=dup_test5)
    f.create_dataset("ques1_length_test", dtype='uint32', data=dup_length_test5)

    f.close()
    print ('h5py file written on ', params['output_h5'])
    
    # create output json file
    out = {}
    out['index_to_word'] = itow # encode the (1-indexed) vocab
    json.dump(out, open(params['output_json'], 'w'))
    print ('json file written on ', params['output_json'])

if __name__ == "__main__":
    
    parser = prepro_parser(maxlen, batch_size)
    args = parser.parse_args()
    params = vars(args) # convert parser arguments to ordinary dict
    print ('parsed input parameters:', json.dumps(params, indent = 2))
    # Run the script
    main(params)
