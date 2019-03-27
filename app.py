from nlp_engine import Nlp

words=["hi","hello","nice","good"]
#print(bag_of_words("hello here nice",words))

n=Nlp(words)
print(n.bag_of_words("hello here nice"))
print(n.vocab)
