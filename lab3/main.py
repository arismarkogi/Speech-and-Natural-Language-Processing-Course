import os
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from early_stopper import EarlyStopper
from attention import SimpleSelfAttentionModel, MultiHeadAttentionModel, TransformerEncoderModel

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN, LSTM
from training import train_dataset, eval_dataset, torch_train_val_split
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50
DATASET = "MR"  # options: "MR", "Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# convert data labels from strings to integers
label_encoder = LabelEncoder()

label_encoder.fit(y_train)

print(y_train[:10])

y_train = label_encoder.transform(y_train)  # EX1
y_test = label_encoder.transform(y_test)  # EX1
n_classes = label_encoder.classes_.size  # EX1 - LabelEncoder.classes_.size

print(y_train[:10])

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)

test_set = SentenceDataset(X_test, y_test, word2idx)

# EX7 - Define our PyTorch-based DataLoader
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) # EX7
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,shuffle=True)  # EX7


#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################

#model = BaselineDNN(output_size= n_classes, embeddings=embeddings, trainable_emb=EMB_TRAINABLE)# EX8
n_head=12
n_layer=12
model = TransformerEncoderModel(output_size=n_classes, embeddings=embeddings, max_length=100, n_head=n_head, n_layer=n_layer)

# move the mode weight to cpu or gpu
model.to(DEVICE)
print(model)

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
criterion = torch.nn.BCEWithLogitsLoss() if n_classes == 2 else torch.nn.CrossEntropyLoss()  # EX8
#parameters = [param for param in model.parameters() if param.requires_grad]  # EX8
parameters = model.parameters()
optimizer = torch.optim.Adam(parameters, lr=0.001)   # EX8

#############################################################################
# Training Pipeline
#############################################################################

# Initialize lists to store metrics
train_accuracy_list = []
train_f1_score_list = []
train_recall_list = []
test_accuracy_list = []
test_f1_score_list = []
test_recall_list = []
val_accuracy_list = []
val_f1_score_list = []
val_recall_list = []

train_loader, val_loader = torch_train_val_split(train_set, BATCH_SIZE, BATCH_SIZE)
save_path = f'./trained_models/best_{model.__class__.__name__}_{DATASET}_embdim{EMB_DIM}.pt'
early_stop = EarlyStopper(model, save_path, patience=5)

num_epochs = 0

for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                            model,
                                                            criterion)
    
    val_loss, (y_val_gold, y_val_pred) = eval_dataset(val_loader,
                                                         model,
                                                         criterion)

    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
                                                         model,
                                                         criterion)
    # Calculate metrics
    train_accuracy = accuracy_score(y_train_gold, y_train_pred)
    train_f1_score = f1_score(y_train_gold, y_train_pred, average='macro')
    train_recall = recall_score(y_train_gold, y_train_pred, average='macro')

    test_accuracy = accuracy_score(y_test_gold, y_test_pred)
    test_f1_score = f1_score(y_test_gold, y_test_pred, average='macro')
    test_recall = recall_score(y_test_gold, y_test_pred, average='macro')

    val_accuracy = accuracy_score(y_val_gold, y_val_pred)
    val_f1_score = f1_score(y_val_gold, y_val_pred, average='macro')
    val_recall = recall_score(y_val_gold, y_val_pred, average='macro')

    # Append metrics to lists
    train_accuracy_list.append(train_accuracy)
    train_f1_score_list.append(train_f1_score)
    train_recall_list.append(train_recall)
    test_accuracy_list.append(test_accuracy)
    test_f1_score_list.append(test_f1_score)
    test_recall_list.append(test_recall)
    val_accuracy_list.append(val_accuracy)
    val_f1_score_list.append(val_f1_score)
    val_recall_list.append(val_recall)

    if early_stop.early_stop(val_loss):
        print('early stopping ...')
        num_epochs = epoch
        break

    model.load_state_dict(torch.load(save_path))


epochs = np.arange(1, int(num_epochs + 1))

# Plotting train and test metrics together for each type of score
plt.figure(figsize=(10, 6))

# Plot Train Accuracy
plt.plot(epochs, train_accuracy_list, label='Train Accuracy', color='blue')
# Plot Test Accuracy
plt.plot(epochs, test_accuracy_list, label='Test Accuracy', color='orange')

plt.plot(epochs, val_accuracy_list, label='Val Accuracy', color='green')


plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title(f'Train, Test, Val Accuracy over Epochs, MR dataset, n_head={n_head}, n_layer={n_layer}')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))

# Plot Train F1-Score
plt.plot(epochs, train_f1_score_list, label='Train F1-Score', color='blue')
# Plot Test F1-Score
plt.plot(epochs, test_f1_score_list, label='Test F1-Score', color='orange')
plt.plot(epochs, val_f1_score_list, label='Val F1-Score', color='green')


plt.xlabel('Number of Epochs')
plt.ylabel('F1-Score')
plt.title(f'Train, Test, Val F1-Score over Epochs, MR dataset, n_head={n_head}, n_layer={n_layer}')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))

# Plot Train Recall
plt.plot(epochs, train_recall_list, label='Train Recall', color='blue')
# Plot Test Recall
plt.plot(epochs, test_recall_list, label='Test Recall', color='orange')
plt.plot(epochs, val_recall_list, label='Val Recall', color='green')


plt.xlabel('Number of Epochs')
plt.ylabel('Recall')
plt.title(f'Train, Test, Val Recall over Epochs, MR dataset, n_head={n_head}, n_layer={n_layer}')
plt.legend()
plt.grid(True)
plt.show()