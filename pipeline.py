from mail import Mail
import pprint
from os import path

# testing run script
# 1. pull emails
# 2. preprocess / clean text
# 3. generate features from text
# 4. stack text vectors into matrix 
# 5. run clustering algorithm 

if path.exists('mail_pickle.pkl'):
  # unpickle 
  print("object has been cached... unpickling")
  mail = Mail.unpickle_obj('mail_pickle.pkl')
  # stack each message vector into matrix
  mail.generate_mail_matrix(True)
  # run models 
  mail.k_means()
  mail.tf_idf(5)
  # do something with output of model(s)
  pprint.pprint(mail.keywords_by_msg)
else:
  mail = Mail()
  # method performs pre-processing on each email text (subject/body)
  messages = mail.get_messages(msg_count=50) # 50 api requests
  # stack each message vector into matrix
  mail.generate_mail_matrix(True)
  # run models 
  mail.k_means()
  mail.tf_idf(5)
  # do something with output of model(s)
  pprint.pprint(mail.keywords_by_msg)
  # pickle object
  Mail.pickle_obj(mail, "mail_pickle.pkl")

# dump results in text file