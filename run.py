import numpy as np
import torch
import torch.nn as nn
import tqdm

NUM_DIGITS = 10
HIDDEN_DIM = 100
BATCH_SIZE = 128

def fizzbuzz_encoder(i):
    if i % 15 == 0:
        return 3
    elif i % 5 == 0:
        return 2
    elif i % 3 == 0:
        return 1
    else:
        return 0

def fizzbuzz_decoder(prediction):
    return [str(prediction), 'fizz', 'buzz', 'fizzbuzz'][prediction]

def helper(i):
    print("%s" % fizzbuzz_decoder(fizzbuzz_encoder(i)))

def binary_encoder(i):
    digits = [i >> d & 1 for d in range(NUM_DIGITS)]    #list
    #return np.array(list(reversed(digits)))
    digits.reverse()    #list
    return np.array(digits) #np.array

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

trX = torch.tensor([binary_encoder(i) for i in range(101, 2**NUM_DIGITS)], dtype=torch.float32, device=device)  #(2**NUM_DIGITS-101) * NUM_DIGITS
trY = torch.tensor([fizzbuzz_encoder(i) for i in range(101, 2**NUM_DIGITS)], dtype=torch.int64, device=device)

model = nn.Sequential(
    nn.Linear(NUM_DIGITS, HIDDEN_DIM),
    nn.ReLU(),
    nn.Linear(HIDDEN_DIM, 4)
).cuda()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10000):
    for start in range(0, len(trX), BATCH_SIZE):
        end = start + BATCH_SIZE
        
        pred = model(trX[start:end])
        loss = loss_fn(pred, trY[start:end])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if not epoch % 100:
        loss = loss_fn(model(trX[start:end]), trY[start:end])
        print("epoch:", epoch, "loss:", loss.item())

#test
testX = torch.tensor([binary_encoder(i) for i in range(1, 100)], dtype=torch.float32, device=device)
with torch.no_grad():
    predX = model(testX)

predictions = list(predX.max(1)[1].data.tolist())

print([fizzbuzz_decoder(prediction) for prediction in predictions])