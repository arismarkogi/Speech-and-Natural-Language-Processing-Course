# Train a torch DNN for Kaldi DNN-HMM model

import math
import sys

import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from dnn.torch_dataset import TorchSpeechDataset
from dnn.torch_dnn import TorchDNN

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device('cpu')
# CONFIGURATION #

NUM_LAYERS = 2
HIDDEN_DIM = 256
USE_BATCH_NORM = True
DROPOUT_P = .2
EPOCHS = 50
PATIENCE = 3

if len(sys.argv) < 2:
    print("USAGE: python timit_dnn.py <PATH/TO/CHECKPOINT_TO_SAVE.pt>")

BEST_CHECKPOINT = sys.argv[1]


# FIXME: You may need to change these paths
TRAIN_ALIGNMENT_DIR = "exp/tri_aligned_train"
DEV_ALIGNMENT_DIR = "exp/tri_aligned_dev"
TEST_ALIGNMENT_DIR = "exp/tri_aligned_test"

def train(model, criterion, optimizer, train_loader, dev_loader, epochs=50, patience=3):
    best_loss = float('inf')
    no_improvement = 0

    for epoch in range(epochs):
        model.train()
        for data, labels in tqdm(train_loader):
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            dev_loss = 0
            dev_samples = 0
            for data, labels in dev_loader:
                data, labels = data.to(DEVICE), labels.to(DEVICE)
                outputs = model(data)
                dev_loss += criterion(outputs, labels).item() * data.size(0)
                dev_samples += data.size(0)
            dev_loss /= dev_samples

        print(f'Epoch {epoch + 1}, Dev Loss: {dev_loss:.4f}')

        if dev_loss < best_loss:
            best_loss = dev_loss
            torch.save(model, BEST_CHECKPOINT)
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print("Stopping early due to no improvement.")
                break



trainset = TorchSpeechDataset('./', TRAIN_ALIGNMENT_DIR, 'train')
validset = TorchSpeechDataset('./', DEV_ALIGNMENT_DIR, 'dev')
testset = TorchSpeechDataset('./', TEST_ALIGNMENT_DIR, 'test')

scaler = StandardScaler()
scaler.fit(trainset.feats)

trainset.feats = scaler.transform(trainset.feats)
validset.feats = scaler.transform(validset.feats)
testset.feats = scaler.transform(testset.feats)

feature_dim = trainset.feats.shape[1]
n_classes = int(trainset.labels.max() - trainset.labels.min() + 1)


dnn = TorchDNN(
    feature_dim,
    n_classes,
    num_layers=NUM_LAYERS,
    batch_norm=USE_BATCH_NORM,
    hidden_dim=HIDDEN_DIM,
    dropout_p=DROPOUT_P
)
dnn.to(DEVICE)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
dev_loader = torch.utils.data.DataLoader(validset, batch_size=128, shuffle=True)

optimizer = torch.optim.Adam(dnn.parameters(), lr=0.001) 
criterion = torch.nn.CrossEntropyLoss()

train(dnn, criterion, optimizer, train_loader, dev_loader, epochs=EPOCHS, patience=PATIENCE)
