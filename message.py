import spacy
import numpy as np
# import torch
import torchtext
import string 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class Message():
  """
  Represents a message (subject, body, sender) data
  """
  def __init__(self, subject, body, sender, dim=50, gen_features=True):
    self.subject = subject
    self.body = body
    self.sender = sender 
    # trained on Wikipedia 2014 corpus; can try other models
    self.dim = dim
    self.glove = torchtext.vocab.GloVe(name="6B",
                              dim=self.dim)

    self.body_feature, self.subject_feature = None, None

    if gen_features:
      self.preprocess()
      self.feature_extraction()

  def _remove_stop_words(self):
    """
    word must be tokenized
    remove common english stopwords
    """
    assert type(self.subject) == list and type(self.body) == list
    stop_words = set(stopwords.words('english'))
    self.subject = [w for w in self.subject if not w in stop_words]
    self.body = [w for w in self.body if not w in stop_words]

  def preprocess(self):
    """
    performs preprocessing steps on raw message
    """
    # lower case
    self.body, self.subject = self.body.lower(), self.subject.lower()
    # remove punctuation
    self.body = self.body.translate(str.maketrans('', '', string.punctuation))
    self.subject = self.subject.translate(str.maketrans('', '',string.punctuation))
    # tokenize
    self.body = nltk.word_tokenize(self.body)
    self.subject = nltk.word_tokenize(self.subject)
    # remove stop words
    self.body = self._remove_stop_words()
    self.subject = self._remove_stop_words()
    # Lemmatize 
    wnl = WordNetLemmatizer()
    self.body = [wnl.lemmatize(w) for w in self.body]
    self.subject = [wnl.lemmatize(w) for w in self.subject]
    # remove non-ascii characters ? 
  
  def get_top_k_words(self, txt, k, corpora):
    """
    returns top K words using TF-IDF algorithm 
    """
    # *** requires building corpora (vocab) of all emails in inbox
    # make corpora a global var (potentially)
    pass 

  def feature_extraction(self):
    """
    method to convert message into feature (vectorize)
    """
    assert type(self.subject) == list and type(self.body) == list
    self.body_feature = np.zeros(self.dim)
    for word in self.body:
      self.body_feature += self.glove(word)
    
    self.subject_feature = np.zeros(self.dim)
    for word in self.subject:
      self.subject_feature += self.glove(word)

  
  def to_string(self):
    """
    """
    pass 