from nlp_engine import Nlp
from model import *
import pickle

class Framework():
    words=list()
    classes=list()
    model=None

    def get_data(self,file):
        with open(file) as json_data:
            data = json.load(json_data)
        return data 

    def prepare_model(self,file):
        # get intents
        data=self.get_data(file)
        # restore all of our data structures
        intents_or_entities=file.split(".")[0]
        data = pickle.load( open( "training_data_%s"%intents_or_entities, "rb" ) )
        words = data['words']
        classes = data['classes']
        train_x = data['train_x']
        train_y = data['train_y']
        # Build neural network
        net = tflearn.input_data(shape=[None, len(train_x[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
        net = tflearn.regression(net)

        # Define model and setup tensorboard
        ann = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
        self.words=words
        self.classes=classes
        self.model=ann

    
    def classify(self,sentence): # Sentence = a pattern = what user can say.
        ERROR_THRESHOLD = 0.4
        self.prepare_model("intents.json")
        # load our saved model
        self.model.load('./model.tflearn')
        n=Nlp(self.words)
        # generate probabilities from the decision tree model
        results = self.model.predict([n.bag_of_words(sentence)])[0]
        print("baggg ",n.bag_of_words(sentence))
        # filter out predictions below a threshold
        results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        """return_list = []
        for r in results:
            return_list.append((classes[r[0]], r[1]))"""
        # return tuple of intent and probability
        return  self.classes[results[0][0]]