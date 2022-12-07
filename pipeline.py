from mail import Mail
import pprint
from os import path
# import argparse

# def cli_parse():
#   parser = argparse.ArgumentParser()

#   #-db DATABSE -u USERNAME -p PASSWORD -size 20
#   parser.add_argument("-db", "--hostname", help="Database name")
#   parser.add_argument("-u", "--username", help="User name")
#   parser.add_argument("-p", "--password", help="Password")
#   parser.add_argument("-size", "--size", help="Size", type=int)

# args = parser.parse_args()


def main():
  """
  1. pull emails
  2. preprocess / clean text
  3. generate features from text
  4. stack text vectors into matrix 
  5. run clustering algorithm 
  """
  if path.exists('mail_pickle.pkl'):
    # unpickle 
    print("object has been cached... unpickling")
    mail = Mail.unpickle_obj('mail_pickle.pkl')
  else:
    mail = Mail()

  # stack each message vector into matrix
  mail.generate_mail_matrix(True)
  # run models 
  mail.k_means()
  mail.tf_idf(5)
  # do something with output of model(s)
  pprint.pprint(mail.clusters)
  # pickle object
  Mail.pickle_obj(mail, "mail_pickle.pkl")


if __name__ == '__main__':
  main() 