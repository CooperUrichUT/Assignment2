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
        self.loss = nn.CrossEntropyLoss()
        self.model = DAN(word_embeddings, word_embeddings.get_embedding_length(), 32, 2)

    def predict(self, ex_words: List[str], has_typos: bool):
        # find the index of each word using the word indexer in the NSC class
        words_idx = []
        for word in ex_words:
            if has_typos:
                word = word[:3]
                
            words_idx.append(max(1, self.word_indexer.index_of(word)))
        # create a torch.tensor of the word indexer, this makes for faster GPU times
        words_tensor=torch.tensor([words_idx])
        # calculate the y_probability using the nn.Module subclass
        y_probabilities = self.model.forward(words_tensor)
        return torch.argmax(y_probabilities)

    def loss(self, probs, target):
        return self.loss(probs, target)
    


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
                                 

    classifier = NeuralSentimentClassifier(word_embeddings)
    word_indices = generate_word_indices(train_exs, classifier, train_model_for_typo_setting)
    ADAM = optim.Adam(classifier.model.parameters(), lr=0.001)
    training_set = np.arange(len(train_exs))
    epochs = 15

    for epoch in range(epochs):
        np.random.shuffle(training_set)
        total_loss = 0.0
        batch_x = []
        batch_y = []
        batch_size = 128

        for idx in training_set:
            if len(batch_x) < batch_size:
                batch_x, batch_y = create_batch(idx, batch_x, batch_y, word_indices, train_exs, padding=50)
            else:
                batch_x, batch_y, total_loss = process_batch(batch_x, batch_y, classifier, ADAM, total_loss)

        if batch_x:
            batch_x, batch_y, total_loss = process_batch(batch_x, batch_y, classifier, ADAM, total_loss)

        total_loss /= len(train_exs)
        print("Total loss on epoch %i: %f" % (epoch, total_loss))

    return classifier

def create_batch(idx, batch_x, batch_y, word_indices, train_exs, padding):
    sent = word_indices[idx]
    sent_pad = np.zeros(padding, dtype=np.int64)
    sent_pad[:min(padding, len(sent))] = sent[:min(padding, len(sent))]
    batch_x.append(sent_pad)
    batch_y.append(train_exs[idx].label)

    return batch_x, batch_y

def process_batch(batch_x, batch_y, classifier, ADAM, total_loss):
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    classifier.model.train()
    ADAM.zero_grad()
    batch_x = torch.tensor(batch_x, dtype=torch.long)
    probs = classifier.model.forward(batch_x)
    target = torch.tensor(batch_y, dtype=torch.long)
    loss = classifier.loss(probs, target)
    total_loss += loss.item()
    loss.backward()
    ADAM.step()
    batch_x = []
    batch_y =[]
    return batch_x, batch_y, total_loss

def generate_word_indices(train_exs, ns_classifier, train_model_for_typo_setting):
    word_indices = {}

    for i in range(len(train_exs)):
        words = train_exs[i].words
        index_list = []
        for word in words:
            if train_model_for_typo_setting:
                word = word[:3]

            idx = ns_classifier.word_indexer.index_of(word)
            index_list.append(max(idx, 1))
        word_indices[i] = index_list

    return word_indices

class DAN(nn.Module):
    def __init__(self, word_embeddings=None, inp=50, hid=32, out=2):
        super(DAN, self).__init__()
        self.embeddings = word_embeddings.get_initialized_embedding_layer() if word_embeddings is not None else None
        self.V = nn.Linear(inp, hid)
        self.g = nn.Tanh()
        self.W = nn.Linear(hid, out)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, index):
        index = self.embeddings(index) if self.embeddings is not None else None
        mean = torch.mean(index, dim=1).float()
        output = self.V(mean)
        output = self.g(output)
        output = self.W(output)
        
        return output
    

class PrefixEmbeddings:
    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_initialized_embedding_layer(self, frozen=True):
        return torch.nn.Embedding.from_pretrained(torch.FloatTensor(self.vectors), freeze=frozen)

    def get_embedding_length(self):
        return len(self.vectors[0])

    def get_embedding(self, word):
        prefix = word[:3]  
        word_idx = self.word_indexer.index_of(prefix)
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[self.word_indexer.index_of("UNK")]
