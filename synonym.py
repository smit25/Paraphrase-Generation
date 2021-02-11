import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
import random

def paraphraseable(tag):
 return tag.startswith('NN') or tag == 'VB' or tag.startswith('JJ')

def pos(tag):
 if tag.startswith('NN'):
  return wn.NOUN
 elif tag.startswith('V'):
  return wn.VERB

def synonyms(word, tag):
  lemma_lists = [ss.lemmas() for ss in wn.synsets(word, pos(tag))]
  lemmas = [lemma.name() for lemma in sum(lemma_lists, [])]
  return set(lemmas)

def subs_synonym(sentence):
  words = word_tokenize(sentence)
  words = pos_tag(words)
  output = []
  for word, tag in words:
    new_word = word
    if paraphraseable(tag):
      syn_set = synonyms(word, tag)
      if syn_set and len(syn_set)>1:
        new_word = random.choice(syn_set)
    output.append(new_word)
  return ' '.join(output)

def antonyms(word, tag):
  antonyms = set()
  flag = False
  for syn in wn.synsets(word, pos=pos):
    for lemma in syn.lemmas():
      for antonym in lemma.antonyms():
         antonyms.add(antonym.name())
  if len(antonyms):
    flag = True

  if flag:
    return antonyms.pop()
  else:
    return None

def subs_antonyms(sentence):
  words = sentence.split(' ')
  l, i = len(words), 0
  tagged_words = pos_tag(words)
  flag = False
  output = []

  while i<l:
    word = words[i]
    new_word = word
    if not flag:
      if word == 'not' and i+1<l:
        antonym = antonyms(words[i+1],tagged_words[i+1][1])
      if antonym != None:
        new_word = antonym
        flag = True
        i+=2
    output.append(new_word)
    i+=1

  return " ".join(output)
    

text = input('Enter the sentence: \n')
# syn_output = subs_synonyms(text)
# ant_output = subs_antonyms(text)