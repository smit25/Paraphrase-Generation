import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn

class Antonym():
  def __init__(self, sentence):

    super(Antonym, self).__init__()
    self.sentence = sentence

  def pos(self, tag):
    if tag.startswith('NN'):
      return wn.NOUN
    elif tag.startswith('V'):
      return wn.VERB

  def antonyms(self, word, tag):
    antonyms = set()
    flag = False
    for syn in wn.synsets(word, pos= self.pos(tag)):
      for lemma in syn.lemmas():
        for antonym in lemma.antonyms():
          antonyms.add(antonym.name())
    if len(antonyms):
      flag = True

    if flag:
      return antonyms.pop()
    else:
      return None

  def main(self):
    words = self.sentence.split(' ')
    l, i = len(words), 0
    tagged_words = pos_tag(words)
    flag = False
    output = []

    while i<l:
      word = words[i]
      new_word = word
      if not flag:
        if word == 'not' and i+1<l:
          antonym = self.antonyms(words[i+1],tagged_words[i+1][1])
          if antonym != None:
            new_word = antonym
            flag = True
            i+=1
      output.append(new_word)
      i+=1

    return " ".join(output)