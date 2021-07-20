from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from torch.hub import import_module
from torch.utils.data import Dataset, dataloader
from torch.utils.data import DataLoader
from torch import nn

class WordEmbeddingDataset(Dataset):
    def __init__(self, text, word_to_idx, word_freqs):
        super(WordEmbeddingDataset, self).__init__
        self.text_encoded = [word_to_idx.get(word, word_to_idx['<unk>']) for word in text]
        self.text_encoded = torch.tensor(self.text_encoded, device=device)
        self.word_freqs = torch.tensor(word_freqs, device=device)
    
    def __len__(self):
        return len(self.text_encoded)
    
    def __getitem__(self, idx):
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx-3, idx)) + list(range(idx, idx+3))
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices]
        neg_words = torch.multinomial(self.word_freqs, 100 * len(pos_words), True) #long tensor, [batch_size, 100*6]
        return center_word, pos_words, neg_words

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embed_in = nn.Embedding(vocab_size, self.embed_size)
        self.embed_out = nn.Embedding(vocab_size, self.embed_size)

    def forward(self, input_labels, pos_labels, neg_labels):
        input_embeddings = self.embed_in(input_labels)  #[batch_size, embed_size]
        pos_embeddings = self.embed_out(pos_labels)     #[batch_size, 6, embed_size]
        neg_embeddings = self.embed_out(neg_labels)     #[batch_size, 6*100, embed_size]

        log_pos = torch.bmm(pos_embeddings, input_embeddings.unsqueeze(2)).squeeze(2)   #[batch_size, 6]
        log_neg = torch.bmm(neg_embeddings, -input_embeddings.unsqueeze(2)).squeeze(2)   #[batch_size, 6*100]
        
        #log_pos = nn.LogSigmoid(log_pos).sum(1)
        #log_neg = nn.LogSigmoid(log_neg).sum(1)
        log_pos = F.logsigmoid(log_pos).sum(1)
        log_neg = F.logsigmoid(log_neg).sum(1)

        loss = log_pos + log_neg

        return -loss

with open('data/text8.train.txt') as f:
    text = f.read()
text = text.split()
text = [word.lower() for word in text]

vocab = dict(Counter(text).most_common(30000))    #max vocab size = 30000
vocab['<unk>'] = len(text) - np.sum(list(vocab.values()))

idx_to_word = [word for word in vocab]
word_to_idx = {word:i for i, word in enumerate(idx_to_word)}

word_counts = np.array([count for count in vocab.values()])
word_freqs = word_counts / len(text)
word_freqs = word_freqs ** (3/4)
word_freqs = word_freqs / np.sum(word_freqs)

vocab_size = len(idx_to_word)

if torch.cuda.is_available:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

dataset = WordEmbeddingDataset(text, word_to_idx, word_freqs)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

model = EmbeddingModel(vocab_size, embed_size=100)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

# train
for epoch in range(2):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()

        loss = model(input_labels, pos_labels, neg_labels).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("epoch: {}, iteration: {}, loss: {}".format(epoch, i, loss.item()))

