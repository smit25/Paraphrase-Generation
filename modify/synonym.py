import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
import spacy_universal_sentence_encoder

nlp = spacy_universal_sentence_encoder.load_model('en_use_lg')

class Synonym():
  def __init__(self, sentence):
    super(Synonym, self).__init__()
    
    self.sentence = sentence
    self.words = []
  
  def tag_true(self,tag):
    return tag.startswith('NN') or tag == 'VB' or tag.startswith('JJ')

  def pos(self,tag):
    if tag.startswith('NN'):
      return wn.NOUN
    elif tag.startswith('V'):
      return wn.VERB

  def synonyms(self, word, tag):
    lemma_lists = [ss.lemmas() for ss in wn.synsets(word, self.pos(tag))]
    lemmas = [lemma.name() for lemma in sum(lemma_lists, [])]
    return set(lemmas)

  def get_best_synonym(self,word, sentence, syn_set, nlp, threshold = 0.96):
    sent1 = nlp(sentence)
    max_score = 0
    result = word
    for candidate in syn_set:
      sent2 = sentence.replace(word,candidate) #replace word by candidate synonym
      sent2 = nlp(sent2)
      score = sent1.similarity(sent2) #compute word mover distance to select the best synonym, we can also use cosine similarity on BERT embedding
      if score > max_score and score < threshold:
        result = candidate
        max_score = score
    
    return result

  def main(self):
    words = word_tokenize(self.sentence)
    words = pos_tag(words)
    output = []
    for word, tag in words:
      new_word = word
      if self.tag_true(tag):
        syn_set = self.synonyms(word, tag)
        if syn_set and len(syn_set)>1:
          new_word = self.get_best_synonym(word, self.sentence, syn_set, nlp)
      output.append(new_word)

    return ' '.join(output)


