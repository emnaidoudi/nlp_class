from nlp_engine import Nlp
import json
import numpy as np
import tflearn
import tensorflow as tf
import random
import pickle

class Model():
    words=list()
    documents=list()
    classes=list()
    train_x=list()
    train_y=list()

    def get_data(self,file):
        with open(file) as json_data:
            data = json.load(json_data)
        return data    

    def extract_vocab(self,file):
        data=self.get_data(file)
        nlp=Nlp()
        words = [] 
        classes = []
        documents = []
        # loop through each sentence in our infos patterns
        for info in data[file.split(".")[0]]:
            for pattern in info['patterns']:
                # tokenize each word in the sentence
                w = nlp.tokenization(pattern)
                # add to our words list
                words.extend(w)
                # add to documents in our corpus
                documents.append((pattern, info['tag']))
                # add to our classes list
                if info['tag'] not in classes:
                    classes.append(info['tag'])   
        self.classes=classes
        self.documents=documents
        self.words=words            
                    
        #return dict({"words":words,"classes":classes,"documents":documents})       


    def train_x_train_y(self,file):
        self.extract_vocab(file)
        nlp=Nlp(self.words)
        # create our training data
        training = []
        # create an empty array for our output
        output_empty = [0] * len(self.classes)
        bag=list()
        # training set, bag of words for each sentence
        d=self.documents
        for doc in d:
            # initialize our bag of words
            bag=nlp.bag_of_words(doc[0])
           
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])

        # shuffle our features and turn into np.array
        random.shuffle(training)
        training = np.array(training)

        # create train and test lists
        self.train_x = list(training[:,0])# the sentence
        self.train_y = list(training[:,1])# the predicted value (label => intent or entity)    


    def train_model(self,file):
        self.train_x_train_y(file)
        # reset underlying graph data
        tf.reset_default_graph()
        # Build neural network
        net = tflearn.input_data(shape=[None, len(self.train_x[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(self.train_y[0]), activation='softmax')
        net = tflearn.regression(net)

        # Define model and setup tensorboard
        model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
        # Start training (apply Gradient Descent algorithm)
        model.fit(self.train_x, self.train_y, n_epoch=1000, batch_size=8, show_metric=True)
        model.save('model.tflearn')  
        # save all of our data structures
        pickle.dump( {'words':self.words, 'classes':self.classes, 'train_x':self.train_x, 'train_y':self.train_y}, open( "training_data_%s"%(file.split(".")[0]), "wb" ) )  


