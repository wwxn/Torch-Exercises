import torch
import torch.nn as nn
import torch.optim as optim
import definition as df
import random
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
test_sentence = "Please send this message to those people who mean something to you,to those who have touched your life in one way or another,to those who make you smile when you really need it,to those that make you see the brighter side of things when you are really down,to those who you want to let them know that you appreciate their friendship.And if you don’t, don’t worry,nothing bad will happen to you,you will just miss out on the opportunity to brighten someone’s day with this message."
test_sentence = test_sentence.split(' ')
print(test_sentence)

test_sentence1 = set(test_sentence)
test_sentence1 = list(test_sentence1)
word_to_idx = {test_sentence1[i]: i for i in range(len(test_sentence1))}
idx_to_word={word_to_idx[key]:key for key,value in word_to_idx.items()}
print(idx_to_word)
model = df.NgramModel(len(word_to_idx), EMBEDDING_DIM, CONTEXT_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),1e-1)

train_input,train_target=df.make_train_data(test_sentence,word_to_idx,CONTEXT_SIZE)
# print(train_target[1],train_input[1])
for i in range(200):
    out=model.forward(train_input)
    loss=criterion.forward(out,train_target)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    # print(loss.data)

model.eval()
test_data=torch.LongTensor([word_to_idx['those'],word_to_idx['people']])
print(test_data)
test_out=model.forward(test_data)
_,test_out=torch.max(test_out,1)
test_out=test_out.numpy()[0]
test_out=idx_to_word[test_out]
print(test_out)