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
    def __init__(self, vocab_size, embedding_dim, hidden_dim, word_embeddings):
        super(NeuralSentimentClassifier, self).__init__()
        self.DAN = DAN(word_embeddings, word_embeddings.get_embedding_length())
        self.embedding = word_embeddings.get_initialized_embedding_layer()
        self.word_indexer = word_embeddings.word_indexer
        self.criterion = nn.NLLLoss()
        self.word_embeddings = word_embeddings
        self.vocab_size = vocab_size
        self.loss = nn.CrossEntropyLoss()
    
    # Setting the model to train (not sure if we need this)    
    # def train(self):
    #     self.DAN.train()
    # # Setting the model to evaluate (not sure if we need this either)
    # def eval(self):
    #     self.DAN.eval()

    def loss_function(self, real, expected):
        return self.loss(real, expected)   
    
    def predict(self, ex_words: List[str], has_typos: bool):
        index_list = [self.word_indexer.index_of(word) or 1 for word in ex_words]
        tensor = torch.tensor(index_list)
        y_probability = self.model.forward(tensor)
        
        # Return the index of the maximum probability for each example in the batch
        return torch.argmax(y_probability, dim=1)
       

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.

    """

    # Data Preprocessing: Convert your training data into a format suitable for training. This may include tokenization, mapping words to indices, and creating batches of data.
    # shuffle data to insure there is no overfitting
    # Define your model
    vocab_size = len(word_embeddings.vectors)
    embedding_dim = word_embeddings.get_embedding_length()
    hidden_dim = 100
    print("Embedding length = ", word_embeddings.get_embedding_length())
    model = NeuralSentimentClassifier(vocab_size, embedding_dim, hidden_dim, word_embeddings)
    ADAM = optim.Adam(model.DAN.parameters(), lr=0.01)
    sentences = [sentence for sentence in train_exs]
    batch_size = 32
    batches = []
    dict_for_index = generate_word_indices(train_exs, model)
    padding = 50

    # Training loop
    for epoch in range(25):
        batch_x = np.empty((0, 50), dtype=np.int32)
        batch_y = np.empty((0,), dtype=np.int32)
        total_loss = 0.0
        for sentence in sentences:
            if len(batch_x) < batch_size:
                batch_x, batch_y = process_sentence(sentence, sentences, dict_for_index, train_exs, batch_x, batch_y, model, sentence.label)
            else:  # len(batch_x) = batch_size
                total_loss = process_batch(model, ADAM, batch_x, batch_y, total_loss)

                # Combine batch_x and batch_y into a tuple
                batches.append((batch_x.numpy(), np.array(batch_y)))

                batch_x = np.empty((0, padding), dtype=np.int32)
                batch_y = np.empty((0,), dtype=np.int32)

        total_loss /= len(train_exs)
        print("Total loss on epoch %i: %f" % (epoch, total_loss))

    return model

def process_batch(model, optimizer, batch_x, batch_y, total_loss):
    model.DAN.train()
    optimizer.zero_grad()
    batch_x = torch.tensor(batch_x)
    probs = model.DAN.forward(batch_x)
    target = torch.tensor(batch_y)
    loss = model.loss(probs, target)
    total_loss += loss
    loss.backward()
    optimizer.step()

    return total_loss

def process_sentence(idx, ex_indices, word_indices, train_exs, batch_x, batch_y, model, label):
    index_list = [max(model.word_indexer.index_of(word), 1) for word in idx.words]
    padding = 50
    sent_pad = np.zeros((padding,), dtype=np.int32)
    # Padding
    sent_pad[:min(padding, len(index_list))] = index_list[:min(padding, len(index_list))]
    batch_x = np.vstack([batch_x, sent_pad])
    

    # Padding
    batch_y = np.append(batch_y, label)
    
    return batch_x, batch_y

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



# Model Initialization: Create an instance of the NeuralSentimentClassifier class, which includes initializing the neural network model, optimizer, and loss function. You'll use the pre-trained word embeddings from word_embeddings to initialize the embedding layer of the model.

# Training Loop: Iterate through your training data in mini-batches. For each batch, perform the following steps:

# a. Forward Pass: Pass the input data (word indices) through your model to get predictions.

# b. Calculate Loss: Compute the loss between the predicted labels and the actual labels.

# c. Backpropagation: Use backpropagation to compute gradients with respect to the loss.

# d. Update Weights: Update the model weights using the optimizer (e.g., stochastic gradient descent) based on the computed gradients.

# Validation: Optionally, evaluate your model on the development set (dev_exs) during training to monitor its performance. You can calculate metrics such as accuracy, precision, recall, or F1 score to assess how well your model is doing.

# Training Termination: Decide when to stop training. This could be based on a fixed number of epochs or a criterion such as early stopping, where training stops if the model's performance on the development set stops improving.

# Return Model: Once training is complete, return the trained NeuralSentimentClassifier model.


class DAN(nn.Module):
    def __init__(self, word_embedding_list, embedding_layers=300):
        super(DAN, self).__init__()
        # Establishing all of the necessary 
        self.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(word_embedding_list.vectors), freeze=False) if word_embedding_list is not None else None
        self.hidden_layers= 32
        self.binary_classifier = 2
        self.linear_layer_1 = nn.Linear(embedding_layers, self.hidden_layers)
        self.linear_layer_2 = nn.Linear(self.hidden_layers, self.binary_classifier)
        self.activation_function = nn.Tanh()
        # weight vectors for predicting and training
        nn.init.xavier_uniform_(self.linear_layer_1.weight)
        nn.init.xavier_uniform_(self.linear_layer_2.weight)

    def forward(self, input_indices):
        input_indices = input_indices.to(torch.float32) 
        if self.word_embeddings is not None:
             return self.linear_layer_2(self.activation_function(self.linear_layer_1(input_indices)))
        
        # predictions are made using arithmatic mean (per slide notes)
        # this will only trigger if the word embedding list is none null
        arithmatic_mean = torch.mean(self.word_embeddings(input_indices), dim=1, keepdim=False).float()
        return self.linear_layer_2(self.activation_function(self.linear_layer_1(arithmatic_mean)))

       


