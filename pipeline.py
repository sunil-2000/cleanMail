from mail import Mail
from message import Message
# testing run script
# 1. pull emails
# 2. preprocess / clean text
# 3. generate features from text
# 4. stack text vectors into matrix 
# 5. run clustering algorithm 

# # # run script
mail = Mail()

# method performs pre-processing on each email text (subject/body)
messages = mail.get_messages(msg_count=50) # 50 api requests

# stack each message vector into matrix
mail.generate_mail_matrix()

# perform kMeans
# mail.k_means()
mail.tf_idf(20)
# print(out)

# dump results in text file


##### NEED TO CREATE METHOD TO PICKLE / CACHE emails , so don't have to 
# run script everytime to get same data 



# # build corpus IF we want to perform Tf-Idf on body text (might be unnecessary)
# because corpus still relatively small 
# corpus = []
# for m in mail.messages:
#   # after each m has been cleaned 
#   corpus.extend(m.body)

# for m in mail.messages:
#   # perform feature extraction 
#   m.feature_extraction(corpus)


# print(mail.messages)


# get emails, test feature eng pipeline, cluster