import torch
from torch.cuda import init
import torchtext
import torch.nn as nn
from torchtext.vocab import Vectors
import numpy as np

class RNNModel(nn.Module):
    def __init__(self, rnn_type, vocab_size, emb_size, hid_size, nlayers=1, dropout=0.5):
        super(RNNModel, self).__init__()
        
        self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, emb_size)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(emb_size, hid_size, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TAN': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("An invalid option for `--model` was supplied, options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']")
            self.rnn = nn.RNN(emb_size, hid_size, nlayers, nonlinearity=nonlinearity, dropout=dropout)

        self.fc = nn.Linear(hid_size, vocab_size)

        self.init_weights()

        self.rnn_type = rnn_type
        self.nlayers = nlayers
        self.hid_size = hid_size

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
    
    def forward(self, input, hidden):
        emb = self.drop(self.embedding(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decode = self.fc(output)

        return decode, hidden

    def init_hidden(self, batch_size, requires_grad=True):
        weights = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weights.new_zeros((self.nlayers, batch_size, self.hid_size), requires_grad=requires_grad),
                    weights.new_zeros((self.nlayers, batch_size, self.hid_size), requires_grad=requires_grad))
        else:
            return weights.new_zeros((self.nlayers, batch_size, self.hid_size), requires_grad=requires_grad)

def repackage_hidden(hidden):
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(repackage_hidden(h) for h in hidden)

def evaluate(model, data):
    model.eval()
    it = iter(data)
    total_loss = 0
    total_count = 0
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad=False)
        for i, batch in enumerate(it):
            data, target = batch.text, batch.target
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            total_count += np.multiply(*data.size())
            total_loss += loss.item()*np.multiply(*data.size())
    
    loss = total_loss / total_count
    model.train()
    return loss

if __name__ == '__main__':
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    BATCH_SIZE = 32
    # EMBEDDING_SIZE = 650
    EMBEDDING_SIZE = 200
    MAX_VOCAB_SIZE = 50000

    TEXT = torchtext.legacy.data.Field(lower=True)
    train, val, test = torchtext.legacy.datasets.LanguageModelingDataset.splits(path='../data/',
        train='text8.train.txt', validation='text8.dev.txt', test='text8.test.txt', text_field=TEXT)
    TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)
    print("vocab size:", len(TEXT.vocab))

    VOCAB_SIZE = len(TEXT.vocab)
    train_iter, val_iter, test_iter = torchtext.legacy.data.BPTTIterator.splits(
        (train, val, test), batch_size=BATCH_SIZE, device=device, bptt_len=32, repeat=False, shuffle=True
    )

    model = RNNModel('LSTM', vocab_size=VOCAB_SIZE, emb_size=EMBEDDING_SIZE, hid_size=EMBEDDING_SIZE, nlayers=2, dropout=0.5)
    if USE_CUDA:
        model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    val_losses = []

    # train
    for epoch in range(2):
        model.train()
        it = iter(train_iter)
        hidden = model.init_hidden(BATCH_SIZE)
        for i, batch in enumerate(it):
            data, target = batch.text, batch.target
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()
            hidden = repackage_hidden(hidden)
            optimizer.zero_grad()
            output, hidden = model(data, hidden)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            # print
            if i % 1000 == 0:
                print("epoch:", epoch, "iter:", i, "loss:", loss.item())
            # evaluate and save
            if i % 10000 == 0:
                val_loss = evaluate(model, val_iter)
                if len(val_losses) == 0 or val_loss < min(val_losses):
                    print("best model, val loss: ", val_loss)
                    torch.save(model.state_dict(), "lm-best.th")
                else:
                    scheduler.step()
                val_losses.append(val_loss)

