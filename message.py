import numpy as np
import torch
import torchtext
import string
import nltk
import re

try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except:
    nltk.download("wordnet")
    nltk.download("stopwords")
    nltk.download("omw-1.4")
    nltk.download("punkt")


class Message:
    """
    Represents a message (subject, body, sender) data

    attributes:
    subject (str): processed/cleaned, subject of message
    raw_subject (str): original uncleaned subject str
    body (str): processed/cleaned, body of message
    sender (str): sender email address of message
    uid (str): uid of email
    keywords (None or list): initialized as None, but can be populated with
    list of keywords of message after calling tfidf from mail object
    glove (torch model): GloVe model used to convert words to vectors
    body_feature (np array): vector that represents body of message
    subject_feature (np array): vector that represents subject of message
    """

    def __init__(self, subject, body, sender, uid, dim=100):
        """
        subject (str): subject string of message
        raw_subject (str): subject string of message
        body (str): body string of message
        sender (str): sender email address of message
        uid (str): uid of email
        dim (int): dimension of glove vector / dimensionality of features generated
        by glove
        """
        self.subject = subject
        self.raw_subject = subject
        self.body = body
        self.sender = sender
        self.uid = uid
        self.keywords = None
        # trained on Wikipedia 2014 corpus; can try other models
        self.glove = torchtext.vocab.GloVe(name="6B", dim=dim)

        self.body_feature, self.subject_feature = None, None
        self.preprocess()
        self.feature_extraction()

    def _remove_stop_words(self):
        """
        helper function for preprocess
        word must be tokenized
        remove common english stopwords
        """
        assert type(self.subject) == list and type(self.body) == list
        stop_words = set(stopwords.words("english"))
        self.subject = [w for w in self.subject if not w in stop_words]
        self.body = [w for w in self.body if not w in stop_words]

    def preprocess(self):
        """
        performs preprocessing steps on raw message
        """
        # lower case
        self.body, self.subject = self.body.lower(), self.subject.lower()
        # remove punctuation
        self.body = self.body.translate(str.maketrans("", "", string.punctuation))
        self.subject = self.subject.translate(str.maketrans("", "", string.punctuation))
        # remove digits
        self.body = re.sub(r"\d+", "", self.body)
        self.subject = re.sub(r"\d+", "", self.subject)
        # tokenize
        self.body = word_tokenize(self.body)
        self.subject = word_tokenize(self.subject)
        # remove unicode chars
        self.body = [(s.encode("ascii", "ignore")).decode("utf-8") for s in self.body]
        self.subject = [
            (s.encode("ascii", "ignore")).decode("utf-8") for s in self.subject
        ]
        self.body = [s for s in self.body if s != ""]
        self.subject = [s for s in self.body if s != ""]
        # remove stop words
        self._remove_stop_words()
        # Lemmatize
        wnl = WordNetLemmatizer()
        self.body = [wnl.lemmatize(w) for w in self.body]
        self.subject = [wnl.lemmatize(w) for w in self.subject]

    def get_top_k_words(self):
        """
        returns top K words using TF-IDF algorithm,
        only returns keywords if tfidf called from mail object
        """
        return self.keywords

    def feature_extraction(self):
        """
        method to convert message into feature (vectorize) using GloVe
        """
        assert type(self.subject) == list and type(self.body) == list

        self.body_feature = self.glove.get_vecs_by_tokens(
            self.body, lower_case_backup=True
        ).numpy()
        self.subject_feature = self.glove.get_vecs_by_tokens(
            self.subject, lower_case_backup=True
        ).numpy()

        self.body_feature = np.sum(self.body_feature, axis=0)
        self.subject_feature = np.sum(self.subject_feature, axis=0)

    def print_closest_words(self, vec, n=5):
        """
        prints closest word to a given GloVe vector
        """
        dists = torch.norm(self.glove.vectors - vec, dim=1)
        lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1])  # sort by distance
        for idx, difference in lst[1 : n + 1]:  # take the top n
            print(self.glove.itos[idx], difference)

    def get_cleaned_body(self):
        """
        return cleaned body of email
        """
        assert type(self.body) == list
        return self.body
