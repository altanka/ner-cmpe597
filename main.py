# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import lstm_encoder_decoder
import numpy as np
import torch.nn.functional as F

from parser import get_dataset

from vae import Dataset, VariationalAutoencoder, vae_train

torch.manual_seed(1)

######################################################################

# lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
# inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

# # initialize the hidden state.
# hidden = (torch.randn(1, 1, 3),
#           torch.randn(1, 1, 3))
# for i in inputs:
#     # Step through the sequence one element at a time.
#     # after each step, hidden contains the hidden state.
#     out, hidden = lstm(i.view(1, 1, -1), hidden)

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
# inputs = torch.cat(inputs).view(len(inputs), 1, -1)
# hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
# out, hidden = lstm(inputs, hidden)
# print(out)
# print(hidden)


######################################################################
# Example: An LSTM for Part-of-Speech Tagging
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In this section, we will use an LSTM to get part of speech tags. We will
# not use Viterbi or Forward-Backward or anything like that, but as a
# (challenging) exercise to the reader, think about how Viterbi could be
# used after you have seen what is going on. In this example, we also refer
# to embeddings. If you are unfamiliar with embeddings, you can read up
# about them `here <https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html>`__.
#
# The model is as follows: let our input sentence be
# :math:`w_1, \dots, w_M`, where :math:`w_i \in V`, our vocab. Also, let
# :math:`T` be our tag set, and :math:`y_i` the tag of word :math:`w_i`.
# Denote our prediction of the tag of word :math:`w_i` by
# :math:`\hat{y}_i`.
#
# This is a structure prediction, model, where our output is a sequence
# :math:`\hat{y}_1, \dots, \hat{y}_M`, where :math:`\hat{y}_i \in T`.
#
# To do the prediction, pass an LSTM over the sentence. Denote the hidden
# state at timestep :math:`i` as :math:`h_i`. Also, assign each tag a
# unique index (like how we had word\_to\_ix in the word embeddings
# section). Then our prediction rule for :math:`\hat{y}_i` is
#
# .. math::  \hat{y}_i = \text{argmax}_j \  (\log \text{Softmax}(Ah_i + b))_j
#
# That is, take the log softmax of the affine map of the hidden state,
# and the predicted tag is the tag that has the maximum value in this
# vector. Note this implies immediately that the dimensionality of the
# target space of :math:`A` is :math:`|T|`.
#
#
# Prepare data:

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# training_data = [
#     # Tags are: DET - determiner; NN - noun; V - verb
#     # For example, the word "The" is a determiner
#     ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
#     ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
# ]
all_data = get_dataset()
X = []
y = []
for (val, label) in all_data:
    X.append(val)
    y.append(label)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
training_data = []
for (index, label) in enumerate(y_train):
    training_data.append((X_train[index], label))
word_to_ix = {}
# For each words-list (sentence) and tags-list in each tuple of training_data
for sent, tags in all_data:
    for word in sent:
        if word not in word_to_ix:  # word has not been assigned an index yet
            # Assign each word with a unique index
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"PERSONB": 0, "PERSONI": 1, "LOCATIONB": 2, "LOCATIONI": 3,
             "ORGANIZATIONB": 4, "ORGANIZATIONI": 5, "O": 6}  # Assign each tag with a unique index

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 100
HIDDEN_DIM = 128

######################################################################
# Create the model:


def windowed_dataset(y, input_window = 5, output_window = 1, stride = 1, num_features = 1):
  
    '''
    create a windowed dataset
    
    : param y:                time series feature (array)
    : param input_window:     number of y samples to give model 
    : param output_window:    number of future y samples to predict  
    : param stide:            spacing between windows   
    : param num_features:     number of features (i.e., 1 for us, but we could have multiple features)
    : return X, Y:            arrays with correct dimensions for LSTM
    :                         (i.e., [input/output window size # examples, # features])
    '''
  
    L = y.shape[0]
    num_samples = (L - input_window - output_window) // stride + 1

    X = np.zeros([input_window, num_samples, num_features])
    Y = np.zeros([output_window, num_samples, num_features])    
    
    for ff in np.arange(num_features):
        for ii in np.arange(num_samples):
            start_x = stride * ii
            end_x = start_x + input_window
            X[:, ii, ff] = y[start_x:end_x, ff]

            start_y = stride * ii + input_window
            end_y = start_y + output_window 
            Y[:, ii, ff] = y[start_y:end_y, ff]

    return X, Y

def numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest):
    '''
    convert numpy array to PyTorch tensor
    
    : param Xtrain:                           windowed training input data (input window size, # examples, # features); np.array
    : param Ytrain:                           windowed training target data (output window size, # examples, # features); np.array
    : param Xtest:                            windowed test input data (input window size, # examples, # features); np.array
    : param Ytest:                            windowed test target data (output window size, # examples, # features); np.array
    : return X_train_torch, Y_train_torch,
    :        X_test_torch, Y_test_torch:      all input np.arrays converted to PyTorch tensors 
    '''
    
    X_train_torch = torch.from_numpy(Xtrain).type(torch.Tensor)
    Y_train_torch = torch.from_numpy(Ytrain).type(torch.Tensor)

    X_test_torch = torch.from_numpy(Xtest).type(torch.Tensor)
    Y_test_torch = torch.from_numpy(Ytest).type(torch.Tensor)
    
    return X_train_torch, Y_train_torch, X_test_torch, Y_test_torch
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

######################################################################
# Train the model:
# x_tensors = []
# y_tensors = []
# for sentence, tags in training_data:
#     idxs_word = [word_to_ix[w] for w in sentence]
#     idxs_tag = [tag_to_ix[w] for w in tags]
#     x_tensors.extend(np.asarray(idxs_word, dtype=np.int32))
#     y_tensors.extend(np.asarray(idxs_tag, dtype=np.int32))

# x_tensors = np.asarray(x_tensors, dtype=np.int32)
# y_tensors = np.asarray(y_tensors, dtype=np.int32)

# iw = 80 
# ow = 20 
# s = 5
# x_tensors, y_tensors = windowed_dataset(y_tensors, input_window = iw, output_window = ow, stride = s)

# x_tensors = torch.from_numpy(x_tensors).type(torch.Tensor)
# y_tensors = torch.from_numpy(y_tensors).type(torch.Tensor)
# train_data = torch.hstack((x_tensors, y_tensors))
# train_loader = torch.utils.data.DataLoader(train_data, batch_size= 784, shuffle=True)
# vae = VariationalAutoencoder(2) # GPU
# vae = vae_train(vae, train_loader)

# model = lstm_encoder_decoder.lstm_seq2seq(
#     input_size=x_tensors.shape[0], hidden_size=15)
# loss = model.train_model(x_tensors, y_tensors, n_epochs=50, target_len=len(tag_to_ix), batch_size=5,
#                          training_prediction='mixed_teacher_forcing', teacher_forcing_ratio=0.6, learning_rate=0.01, dynamic_tf=False)

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
print(model)
# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)

for epoch in range(20):  # again, normally you would NOT do 300 epochs, it is toy data
    print("Epoch: %d" % (epoch + 1))
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    # inputs = prepare_sequence(training_data[0][0], word_to_ix)
    # tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    # line = training_data[0][0]
    # sentence_tag_list = []
    # for (index, score) in enumerate(tag_scores):
    #     max_val = max(score)
    #     max_ind = list(score).index(max_val)
    #     tag = (list(tag_to_ix.keys())[max_ind])
    #     print(line[index], tag)
    # print(tag_scores)
    out = """<style type="text/css" media="screen">
        table {
            border-collapse: collapse;
            border: 1px solid black;
            font-size: 12px;
        }

        table td {
            border: 1px solid black;
        }
        </style>"""
    out += '<table>' + '<tbody>'
    total_accuracy = 0
    for (index, value) in enumerate(X_test):
        inputs = prepare_sequence(value, word_to_ix)
        correct_tags = y_test[index]
        tag_scores = model(inputs)
        predicted_tags = []
        for (index, score) in enumerate(tag_scores):
            max_val = max(score)
            max_ind = list(score).index(max_val)
            predicted_tags.append(max_ind)
        diff_count = sum(map(lambda x, y: bool(
            x-tag_to_ix[y]), predicted_tags, correct_tags))
        accuracy = 1 - diff_count / len(predicted_tags)
        total_accuracy += accuracy
        words_line = ''
        correct_line = ''
        predicted_line = ''
        for (word_ind, word) in enumerate(value):
            words_line += '<td>' + word + '</td>'
            correct_line += '<td>' + correct_tags[word_ind] + '</td>'
            predicted_line += '<td>' + \
                (list(tag_to_ix.keys())[predicted_tags[word_ind]]) + '</td>'
        out += '<tr>'
        out += words_line
        out += '</tr>'
        out += '<tr>'
        out += correct_line
        out += '</tr>'
        out += '<tr>'
        out += predicted_line
        out += '</tr>'
    final_accuracy = total_accuracy / len(X_test)
    out += '</tbody></table>'
    print('Accuracy: %f' % final_accuracy)
    with open('output.html', 'w') as f:
        f.write(out)



