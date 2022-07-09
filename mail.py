import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import numpy as np
from sklearn.cluster import KMeans
import pprint 

from message import Message


class Mail():
    """
    Represents a users mail account
    """

    def __init__(self, cached_mail=[], cached=False):
        if not cached:
            # If modifying these scopes, delete the file token.json.
            SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

            creds = None
            if os.path.exists('token.json'):
                creds = Credentials.from_authorized_user_file('token.json', SCOPES)
                # If there are no (valid) credentials available, let the user log in.
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                    creds = flow.run_local_server(port=0)
                # Save the credentials for the next run
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())
            
            self.service = build('gmail', 'v1', credentials=creds)
        self.messages = [] if not cached else cached_mail

    def get_messages(self, msg_count=50):
        inbox = self.service.users().messages().list(userId="me", maxResults=msg_count).execute()
        inbox = inbox.get('messages')

        for msg in inbox:
        # Get the message from its id; txt is JSON or dict
            txt = self.service.users().messages().get(userId='me', id=msg['id']).execute()

            try:
                # Get value of 'payload' from dictionary 'txt'
                payload = txt['payload']
                headers = payload['headers']
                
                # Look for Subject and Sender Email in the headers
                for d in headers:
                    if d['name'] == 'Subject':
                        subject = d['value']
                    if d['name'] == 'From':
                        sender = d['value']
                
                # snippet is short part of the message text.
                # we can revise later if we want more body data (but might
                # just add noise)
                body = txt['snippet'] 

                # append to message to messages list 
                self.messages.append(Message(subject, body, sender, msg['id']))
            except:
                print('missing field from message')
                pass
    
    def generate_mail_matrix(self, dump_csv=False):
        """
        generates two matrices: one for all body vectors (B); one for all subject
        vectors (S); both are NxM where N is number of emails and M is dimension
        of GloVe vector used to encode words
        (used for prior to clustering algo)
        """
        self.id_labels = [m.uid for m  in self.messages]
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

        km = KMeans(n_clusters=5).fit(self.B)
        print(km.labels_)

        clusters = {}
        for i in range(len(km.labels_)):
            label = km.labels_[i]
            if label not in clusters:
                clusters[label] = labels[i]
            else:
                clusters[label].append(labels[i])
        
        pprint.pprint(clusters)
