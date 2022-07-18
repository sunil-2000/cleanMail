import numpy as np
import torch
import torchtext
import string 
import nltk 

try:
  from nltk.tokenize import word_tokenize
  from nltk.corpus import stopwords
  from nltk.stem import WordNetLemmatizer
except:
  nltk.download('wordnet')
  nltk.download('stopwords')
  nltk.download('omw-1.4')
  nltk.download('punkt')

class Message():
  """
  Represents a message (subject, body, sender) data
  """
  def __init__(self, subject, body, sender, uid, dim=100, gen_features=True):
    self.subject = subject
    self.body = body
    self.sender = sender 
    self.uid = uid
    # trained on Wikipedia 2014 corpus; can try other models
    self.dim = dim
    self.glove = torchtext.vocab.GloVe(name="6B",
                              dim=dim)

    self.body_feature, self.subject_feature = None, None

    if gen_features:
      self.preprocess()
      self.feature_extraction()

  def _remove_stop_words(self) -> None:
    """
    word must be tokenized
    remove common english stopwords
    """
    assert type(self.subject) == list and type(self.body) == list
    stop_words = set(stopwords.words('english'))
    self.subject = [w for w in self.subject if not w in stop_words]
    self.body = [w for w in self.body if not w in stop_words]

  def preprocess(self) -> None:
    """
    performs preprocessing steps on raw message
    """
    # lower case
    self.body, self.subject = self.body.lower(), self.subject.lower()
    # remove punctuation
    self.body = self.body.translate(str.maketrans('', '', string.punctuation))
    self.subject = self.subject.translate(str.maketrans('', '',string.punctuation))
    # tokenize
    self.body = word_tokenize(self.body)
    self.subject = word_tokenize(self.subject)
    # remove stop words
    # print(type(self.body) == list, type(self.subject) == list)
    self._remove_stop_words()
    # Lemmatize 
    wnl = WordNetLemmatizer()
    self.body = [wnl.lemmatize(w) for w in self.body]
    self.subject = [wnl.lemmatize(w) for w in self.subject]
    # remove non-ascii characters ? 
  
  def get_top_k_words(self, txt, k, corpus):
    """
    returns top K words using TF-IDF algorithm 
    """
    # *** requires building corpora (vocab) of all emails in inbox
    # make corpora a global var (potentially)
    pass 

  def feature_extraction(self) -> None:
    """
    method to convert message into feature (vectorize)
    corpus: full vocabulary of email inbox 
    """
    assert type(self.subject) == list and type(self.body) == list

    self.body_feature = self.glove.get_vecs_by_tokens(self.body, lower_case_backup=True).numpy()
    self.subject_feature = self.glove.get_vecs_by_tokens(self.subject, lower_case_backup=True).numpy()
    
    self.body_feature = np.sum(self.body_feature, axis=0)
    self.subject_feature = np.sum(self.subject_feature, axis=0)

  def print_closest_words(self, vec, n=5):
    dists = torch.norm(self.glove.vectors - vec, dim=1) 
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1]) # sort by distance
    for idx, difference in lst[1:n+1]: 	# take the top n
        print(self.glove.itos[idx], difference)

  def get_cleaned_body(self) -> list:
    """
    return cleaned body of email
    """
    assert type(self.body) == list
    return self.body
