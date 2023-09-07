# models.py
# https://www.youtube.com/watch?v=losFCNJbnZY&ab_channel=JordanBoyd-Graber

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, word_embeddings: WordEmbeddings):
        SentimentClassifier.__init__(self)
        self.word_indexer = word_embeddings.word_indexer
        self.input = word_embeddings.get_embedding_length()
        self.hidden= 32
        self.output= 2
        self.loss = nn.CrossEntropyLoss()
        self.model = DAN(word_embeddings, self.input, self.hidden, self.output)

    def predict(self, ex_words: List[str], has_typos: bool):
        # find the index of each word using the word indexer in the NSC class
        words_idx = [max(1, self.word_indexer.index_of(word)) for word in ex_words]
        # create a torch.tensor of the word indexer, this makes for faster GPU times
        words_tensor=torch.tensor([words_idx])
        # calculate the y_probability using the nn.Module subclass
        y_probability = self.model.forward(words_tensor)
        return torch.argmax(y_probability)

    def loss(self, probs, target):
        return self.loss(probs, target)
    


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
                                 

    classifier = NeuralSentimentClassifier(word_embeddings)
    word_indices = generate_word_indices(train_exs, classifier)
    ADAM = optim.Adam(classifier.model.parameters(), lr=0.001)
    training_set = np.arange(len(train_exs))  # Use NumPy array

    for epoch in range(15):
        np.random.shuffle(training_set)  # Shuffle NumPy array
        total_loss = 0.0
        batch_x = []
        batch_y = []
        padding = 50

        for idx in training_set:
            if len(batch_x) < 128:
                batch_x, batch_y = create_batch(idx, batch_x, batch_y, word_indices, train_exs, padding)
            else:
                batch_x, batch_y, total_loss = process_batch(batch_x, batch_y, classifier, ADAM, total_loss)

        if batch_x:
            batch_x, batch_y, total_loss = process_batch(batch_x, batch_y, classifier, ADAM, total_loss)

        total_loss /= len(train_exs)
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    
    return classifier


def create_batch(idx, batch_x, batch_y, word_indices, train_exs, padding):
    sent_pad = np.zeros(padding, dtype=np.int64)
    sent = word_indices[idx]
    sent_pad[:min(padding, len(sent))] = sent[:min(padding, len(sent))]
    batch_x.append(sent_pad)
    y = train_exs[idx].label
    batch_y.append(y)
    return batch_x, batch_y

def process_batch(batch_x, batch_y, classifier, ADAM, total_loss):
    classifier.model.train()
    ADAM.zero_grad()
    batch_x = torch.tensor(batch_x)
    probs = classifier.model.forward(batch_x)
    target = torch.tensor(batch_y)
    loss = classifier.loss(probs, target)
    total_loss += loss.item()
    loss.backward()
    ADAM.step()
    batch_x = []
    batch_y = []
    return batch_x, batch_y, total_loss

def generate_word_indices(train_exs, ns_classifier):
    word_indices = {}

    for i in range(len(train_exs)):
        words = train_exs[i].words
        index_list = []
        for word in words:
            idx = ns_classifier.word_indexer.index_of(word)
            index_list.append(max(idx, 1))
        word_indices[i] = index_list

    return word_indices

class DAN(nn.Module):
    def __init__(self, word_embeddings=None, inp=50, hid=32, out=2):
        super(DAN, self).__init__()
        if word_embeddings is not None:
            vocab = len(word_embeddings.vectors)
            self.embeddings =nn.Embedding.from_pretrained(torch.from_numpy(word_embeddings.vectors), freeze=False)
        self.V = nn.Linear(inp, hid)
        self.g = nn.Tanh()
        #self.g = nn.ReLU()
        self.W = nn.Linear(hid, out)
        #self.log_softmax = nn.LogSoftmax(dim=0)
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x):
        if self.embeddings is not None :
            word_embedding = self.embeddings(x) 
            mean = torch.mean(word_embedding, dim=1, keepdim=False).float()
            return self.W(self.g(self.V(mean)))
        else:
            return self.W(self.g(self.V(x)))

       


