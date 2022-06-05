# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split

from parser import get_dataset


torch.manual_seed(1)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


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

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
print(model)
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)

for epoch in range(1):  # again, normally you would NOT do 300 epochs, it is toy data
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



