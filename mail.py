import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pprint
import pickle

from message import Message
from credentials import GoogleAuth


class Mail(GoogleAuth):
    """
    Represents a users mail account

    attributes:
    messages (list): list of Message objects from user inbox
    keywords_by_msg (list): list of dictionaries where each dictionary
    contains message id, message raw subject, and keywords from message
    keywords (list): list of keywords derived from set of messages in inbox
    clusters (dict): dictionary where each key is a cluster number that maps
    to a list of messages in the cluster

    to generate messages:
    (1) get_messages

    to generate clusters:
    (1) get_messages
    (2) generate_mail_matrix
        (2a) if using tfidf as features instead of glove vectors, call tf_idf
    (3) kmeans

    to generate keywords:
    (1) get_messages
    (2) tf_idf
    """

    def __init__(self):
        super().__init__()
        self.messages = []
        self.keywords_by_msg = []
        self.keywords = []
        self.clusters = {}
        self.B = None # matrix of all email bodies vectorized
        self.S = None # matrix of all email subjects vectorized

    @staticmethod
    def pickle_obj(object, file_name):
        """
        class method that pickles mail object
        """
        f = open(file_name, "wb")
        pickle.dump(object, f)
        f.close()

    @staticmethod
    def unpickle_obj(file_name):
        """
        if object has been pickled, unpickles and loads the object
        file_name (str): file name of pickled object
        """
        f = open(file_name, "rb")
        mail = pickle.load(f)
        f.close()
        return mail

    def get_messages(self, msg_count=50):
        inbox = (
            self.service.users()
            .messages()
            .list(userId="me", maxResults=msg_count)
            .execute()
        )
        inbox = inbox.get("messages")

        for msg in inbox:
            # Get the message from its id; txt is JSON or dict
            txt = (
                self.service.users().messages().get(userId="me", id=msg["id"]).execute()
            )

            try:
                # Get value of 'payload' from dictionary 'txt'
                payload = txt["payload"]
                headers = payload["headers"]

                # Look for Subject and Sender Email in the headers
                for d in headers:
                    if d["name"] == "Subject":
                        subject = d["value"]
                    if d["name"] == "From":
                        sender = d["value"]

                # snippet is short part of the message text.
                # we can revise later if we want more body data (but might
                # just add noise)
                body = txt["snippet"]
                # append to message to messages list
                self.messages.append(Message(subject, body, sender, msg["id"]))
            except Exception as e:
                print(e)
                pass

    def generate_mail_matrix(self, dump_csv=False):
        """
        generates two matrices: one for all body vectors (B); one for all subject
        vectors (S); both are NxM where N is number of emails and M is dimension
        of GloVe vector used to encode words
        (used for prior to clustering algo)

        dump_csv (bool): whether to dump matrices into csv files (for reuse)
        """
        self.desc_labels = [m.raw_subject for m in self.messages]
        self.B = np.vstack([m.body_feature for m in self.messages])
        self.S = np.vstack([m.subject_feature for m in self.messages])

        if dump_csv:
            np.savetxt("B.csv", self.B, delimiter=",")
            np.savetxt("S.csv", self.S, delimiter=",")

    def k_means(self, desc_labels=True, pp=False, k=4, tf_idf=False):
        """
        performs k means clustering on set of emails

        desc_labels (bool): whether to include full message object or just
        raw subject of message into cluster
        or id of messages (uid)
        pp (bool): whether to pretty print emails by cluster
        k (int): number of clusters
        tf_idf (bool): whether to use tf_idf or glove vector as feature
        performs clustering on inbox
        """
        data = self.S if not tf_idf else self.tfidf_matrix
        labels = self.messages if desc_labels else self.desc_labels

        km = KMeans(n_clusters=k).fit(data)
        # km.labels_ array where each element corresponds to row in self.B
        #  matrix 0th element -> 0th row
        self.emails_by_cluster = km.labels_

        # value of each element im km.labels_ is cluster number assingment
        clusters = {}
        for i in range(len(km.labels_)):
            cluster = km.labels_[i]
            if cluster not in clusters:
                clusters[cluster] = [labels[i]]
            else:
                clusters[cluster].append(labels[i])

        self.clusters = clusters
        if pp:
            pprint.pprint(clusters)

    def tf_idf(self, k):
        """
        using tf-idf algorithm to extract keywords for each msg in inbox
        and generate 20 keywords for user inbox

        k (int): number of keywords to generate per email
        """
        # each element of list is email message body string
        inbox_msg_bodies = [" ".join(msg.body) for msg in self.messages]
        # column represents word in vocab, row represents email in inbox (nsamples x nfeatures)
        tfidf_vector = TfidfVectorizer(
            max_df=0.80, max_features=1000, stop_words="english"
        )
        inbox_matrix = tfidf_vector.fit_transform(inbox_msg_bodies).toarray()
        self.tfidf_matrix = inbox_matrix
        features = tfidf_vector.get_feature_names_out()

        for i in range(len(inbox_matrix)):
            row = inbox_matrix[i]
            top_features = self._extract_top_k_words(k, row, features)
            top_words = [feature[0] for feature in top_features]
            msg_key_words = {
                "uid": self.messages[i].uid,
                "subject": self.messages[i].raw_subject,
                "keywords": top_words,
            }
            self.messages[i].top_k_words = top_words
            self.keywords_by_msg.append(msg_key_words)

        self.keywords = self._k_inbox_keywords_index(inbox_matrix, 20, features)

    def _extract_top_k_words(self, k, vector, features):
        """
        returns top k words from tf-idf word vector representation
        list of (word, score) sorted in descending order by score

        k (int): number of keywords
        vector (np array): tfidf feature representation of a message
        features (np array): array where each element is a string and the index
        corresponds to a feature (e.g., 0th element -> is the 0th tfidf feature)
        """
        top_ids = np.argsort(vector)[::-1][:k]
        top_feats = [(features[i], vector[i]) for i in top_ids if vector[i] != 0]
        return top_feats

    def _k_inbox_keywords_index(self, a, k, features):
        """
        from 2D numpy matrix (a)
        (1) gets largest k indeces [i,j] (ie have the greatest tfidf score)
        (2) as matrix is n_samples x n_features, can map indeces back to words
        s.t a[i,j] -> features[j]
        (3) return top k unique features using this mapping

        a (np 2D array): nsample x nfeatures array
        k (int): number of keywords
        features (np array): array where each element is a string and the index
        corresponds to a feature (e.g., 0th element -> is the 0th tfidf feature)
        """
        idx = np.argsort(a.ravel())[: -k - 1 : -1]
        idx_lst = np.column_stack(np.unravel_index(idx, a.shape))

        keywords = []
        for pos in idx_lst:
            i, j = pos
            word = features[j]
            if word not in keywords:
                keywords.append(word)

        return keywords
