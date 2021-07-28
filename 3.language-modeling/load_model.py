from run import BATCH_SIZE, EMBEDDING_SIZE, RNNModel, TEXT, USE_CUDA, VOCAB_SIZE, evaluate, device
import torch
import torchtext
import numpy as np

best_model = RNNModel('LSTM', VOCAB_SIZE, EMBEDDING_SIZE, EMBEDDING_SIZE, 2, dropout=0.5)
if USE_CUDA:
    best_model.cuda()
best_model.load_state_dict(torch.load('lm-best.th'))

train, val, test = torchtext.legacy.datasets.LanguageModelingDataset.splits(path='../data/',
    train='text8.train.txt', validation='text8.dev.txt', test='text8.test.txt', text_field=TEXT)
train_iter, val_iter, test_iter = torchtext.legacy.data.BPTTIterator.splits(
    (train, val, test), batch_size=BATCH_SIZE, device=device, bptt_len=32, repeat=False, shuffle=True
)

val_loss = evaluate(best_model, val_iter)
print("perplexity on val set:", np.exp(val_loss))

test_loss = evaluate(best_model, test_iter)
print("perplexity on test set:", np.exp(test_loss))



