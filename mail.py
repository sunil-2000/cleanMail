import numpy as np
from sklearn.cluster import KMeans
import pprint

from message import Message
from credentials import GoogleAuth


class Mail(GoogleAuth):
    """
    Represents a users mail account
    """

    def __init__(self, cached_mail=[], cached=False):
        super().__init__()
        self.messages = [] if not cached else cached_mail

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
        """
        self.id_labels = [m.uid for m in self.messages]
        self.desc_labels = [m.subject for m in self.messages]
        self.B = np.vstack([m.body_feature for m in self.messages])
        self.S = np.vstack([m.subject_feature for m in self.messages])

    def k_means(self, desc_labels=True):
        """
        desc_labels(bool): include descriptive labels (subject)
        or id of messages (uid)
        performs clustering on inbox
        """
        # if not self.B and not self.S:
        #     print("need to generate mail matrix")
        #     return
        labels = self.desc_labels if desc_labels else self.id_labels

        km = KMeans(n_clusters=4).fit(self.S)
        print(km.labels_)
        # km.labels_ array where each element corresponds to row in self.B matrix 0th element -> 0th row
        # value of each element im km.labels_ is cluster number assingment
        clusters = {}
        for i in range(len(km.labels_)):
            cluster = km.labels_[i]
            if cluster not in clusters:
                clusters[cluster] = labels[i]
            else:
                clusters[cluster].append(labels[i])

        pprint.pprint(clusters)
