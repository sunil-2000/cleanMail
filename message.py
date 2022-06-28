from distutils.log import WARN

class Message():
  """
  Represents a message (subject, body, sender) data
  """
  def __init__(self, subject, body, sender):
    self.subject = subject
    self.body = body
    self.sender = sender 

  def feature_extraction(self):
    """
    method to convert message into feature (vectorize)
    """
    WARN("not implemented")