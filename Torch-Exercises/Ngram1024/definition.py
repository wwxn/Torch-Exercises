import torch.nn as nn
import torch

class NgramModel(nn.Module):
    def __init__(self, word_number, dim, context_size):
        super(NgramModel, self).__init__()
        self.context_size=context_size
        self.dim=dim
        self.embedding = nn.Embedding(word_number, dim)
        self.linear = nn.Sequential(nn.Linear(context_size * dim, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, word_number),
                                    nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.embedding(x)
        # print(x)
        x = x.reshape(-1, self.context_size*self.dim)
        x = self.linear(x)
        # print(x)
        return x


def make_train_data(sentence, idx, context_size):
    train_input = []
    train_target = []
    for i in range(len(sentence) - context_size):
        train_data = [idx[sentence[i + t]] for t in range(context_size)]
        train_output = idx[sentence[i + context_size]]
        train_input.append(train_data)
        train_target.append(train_output)
    return torch.LongTensor(train_input), torch.LongTensor(train_target)
