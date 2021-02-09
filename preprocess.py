"""
Preprocessing a raw json dataset into json/hdf5 files.
Experimental protocols mentioned in (Gupta et al., 2017) for the Quora dataset are followed
This script will generate:
- quora_data_prepro.h5 
- quora_data_prepro.json

"""
import copy
from random import shuffle, seed
import sys
import os.path
import argparse
import glob
import numpy as np
from scipy.misc import imread, imresize
import scipy.io
import pdb
import string
import h5py
from nltk.tokenize import word_tokenize
import json
import re
import math

maxlen = 26
batch_size = 32
# Tokenize the question
def tokenize(sentence, params):
    if params['token_method'] == 'nltk':
        return(word_tokenize(sentence))
    else:
        return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n']

# Preprocess all the questions and duplicates
def prepro_question(imgs, params, c=0):
    for i,img in enumerate(imgs):
        # c represents the original or duplicate question in every row
        if c==0:
            s = img['question']
            txt = tokenize(s, params)
            img['processed_tokens'] = txt
        elif c==1:
            s = img['question'+str(c)]
            txt = tokenize(s, params)
            img['processed_tokens_duplicate'] = txt

        if i < 10: print (txt)
        if i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(imgs), i*100.0/len(imgs)) )
            sys.stdout.flush()   
    return imgs

# Replacing less frequent words with 'UNK' and  build vocabulary for question and duplicates.
def build_vocab_question(imgs5, params):
 
    count_thr = params['word_count_threshold']
    counts = {} # word vs its frequency
    
    #count words for every original question in every row
    for img in imgs5:
        for w in img['processed_tokens']:
            counts[w] = counts.get(w, 0) + 1

    cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    print ('Top words and their counts:', '\n'.join(map(str,cw[:20])))

    # print obtained results
    total_words = sum(counts.itervalues())
    print ('total words:', total_words)

    # segregte words on the basis of their frequency relative to the threshold
    low_words = [w for w,n in counts.iteritems() if n <= count_thr]
    vocab = [w for w,n in counts.iteritems() if n > count_thr]   
    low_count = sum(counts[w] for w in low_words)

    print ('number of bad words: %d/%d = %.2f%%' % (len(low_words), len(counts), len(low_words)*100.0/len(counts)))
    print ('number of words in vocab would be %d' % (len(vocab), ))
    print ('number of UNKs: %d/%d = %.2f%%' % (low_count, total_words, low_count*100.0/total_words))

    # Words with less frequency mapped to 'UNK'
    print ('inserting the special UNK token')
    vocab.append('UNK')
  
    for img in imgs5:
        txt = img['processed_tokens']
        question = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
        img['final_question'] = question
        txt_c = img['processed_tokens_caption']
        duplicate = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt_c]
        img['final_duplicate'] = duplicate
    
    return imgs5, vocab

# Applying vocab on validation and test sets for question and duplicate.
def apply_vocab_question(imgs, wtoi):  
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if wtoi.get(w,len(wtoi)+1) != (len(wtoi)+1) else 'UNK' for w in txt]
        img['final_question'] = question
        txt_c = img['processed_tokens_duplicate']
        duplicate = [w if w in wtoi else 'UNK' for w in txt_c]
        img['final_duplicate'] = duplicate 

    return imgs

# Embedding the questions into one hot vector using Bag of Words Model (word2vec)
def encode_question(imgs, params, wtoi):

    max_length = params['max_length'] # of a question
    N = len(imgs)

    # For storing encoded question qords
    label_arrays = np.zeros((N, max_length), dtype='uint32')
    label_length = np.zeros(N, dtype='uint32')
    question_id = np.zeros(N, dtype='uint32')
    question_counter = 0
    
    # Storing encoded duplicate words
    duplicate_arrays = np.zeros((N, max_length), dtype='uint32') 
    duplicate_length = np.zeros(N, dtype='uint32')
       
    for i,img in enumerate(imgs):
        question_id[question_counter] = img['id'] #unique_id
        label_length[question_counter] = min(max_length, len(img['final_question'])) # record the length of this question sequence
        duplicate_length[question_counter] = min(max_length, len(img['final_duplicate'])) # record the length of this duplicate sequence        
        question_counter += 1
        for k,w in enumerate(img['final_question']):
            if k < max_length:
                label_arrays[i,k] = wtoi[w]
        for k,w in enumerate(img['final_duplicate']):
            if k < max_length:
                caption_arrays[i,k] = wtoi[w]            
  
    return label_arrays, label_length, question_id, duplicate_arrays, duplicate_length

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
    imgs_train5,vocab = build_vocab_question(imgs_train5, params)
    itow = {i+1:w for i,w in enumerate(vocab)} # 1-indexed vocab translation table
    wtoi = {w:i+1 for i,w in enumerate(vocab)} # Bag of words model

    ques_train5, ques_length_train5, question_id_train5 , cap_train5, cap_length_train5 = encode_question(imgs_train5, params, wtoi)
    
    imgs_test5 = apply_vocab_question(imgs_test5, wtoi)
    
    ques_test5, ques_length_test5, question_id_test5 , cap_test5, cap_length_test5 = encode_question(imgs_test5, params, wtoi)
    
    # Creating and writing on the h5 file
    N = len(imgs_train5)
    f = h5py.File(params['output_h5'], "w")

    ## for train data
    f.create_dataset("ques_train", dtype='uint32', data=ques_train5)
    f.create_dataset("ques_length_train", dtype='uint32', data=ques_length_train5)
    f.create_dataset("ques_cap_id_train", dtype='uint32', data=question_id_train5)# ques_cap_id
    f.create_dataset("ques1_train", dtype='uint32', data=cap_train5)
    f.create_dataset("ques1_length_train", dtype='uint32', data=cap_length_train5)

    ## for test data
    f.create_dataset("ques_test", dtype='uint32', data=ques_test5)
    f.create_dataset("ques_length_test", dtype='uint32', data=ques_length_test5)
    f.create_dataset("ques_cap_id_test", dtype='uint32', data=question_id_test5)
    f.create_dataset("ques1_test", dtype='uint32', data=cap_test5)
    f.create_dataset("ques1_length_test", dtype='uint32', data=cap_length_test5)

    f.close()
    print ('written on ', params['output_h5'])
    
    # create output json file
    out = {}
    out['index_to_word'] = itow # encode the (1-indexed) vocab
    json.dump(out, open(params['output_json'], 'w'))
    print ('wrote ', params['output_json'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input and output jsons and h5 on terminal ( defining the params)
    parser.add_argument('--input_train_json5', default='../data/quora_raw_train.json', help='input json file to process into hdf5')
    parser.add_argument('--input_test_json5', default='../data/quora_raw_test.json', help='input json file to process into hdf5')
    parser.add_argument('--output_json', default='../data/quora_data_prepro.json', help='output json file')
    parser.add_argument('--output_h5', default='../data/quora_data_prepro.h5', help='output h5 file')
    # parser.add_argument('--num_ans', default=1000, type=int, help='number of top answers for the final classifications.')

    # Parameters
    parser.add_argument('--max_length', default = maxlen, type=int, help='max length of a question, in number of words.')
    parser.add_argument('--word_count_threshold', default=0, type=int, help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--token_method', default='nltk', help='tokenization method.')    
    parser.add_argument('--batch_size', default=batch_size, type=int)
    # parser.add_argument('--num_test', default=0, type=int, help='number of test images (to withold until very very end)')

    args = parser.parse_args()
    params = vars(args) # convert parser arguments to ordinary dict
    print ('parsed input parameters:', json.dumps(params, indent = 2))
    # Run the script
    main(params)
