from model import Model
from nlp_engine import *
from framework import Framework

"""n=Nlp(["hi","there","nice","meet","to"])
print(n.bag_of_words("hi there nice to meet "))
print(n.vocab)

m=Model()"""
#m.train_model("intents.json")

f=Framework()
print(f.classify("i want to rent a moped"))
