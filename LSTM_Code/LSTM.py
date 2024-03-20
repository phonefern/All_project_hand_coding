import numpy
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pylab as plt
from tqdm import tqdm
import os
from alive_progress import alive_bar
from time import sleep
from datetime import datetime

torch.backends.cudnn.enabled = False
torch.cuda.empty_cache()

# x = os.listdir('./data/ThermalCameraRaw')

# test_train_files = x.copy()
# test_train_files.pop(index)
# training_files = test_train_files
# testing_files = [filetest]

# training_files = ['jagun.csv', 'yui.csv', 'buk.csv']
# testing_files = ['nan.csv']
training_files = ['train_all_1_11.csv']
# testing_files = ['Train_all.csv']
testing_files = ['train_all_1_11_split.csv']

print(" --------------------- STEP 1: LOADING DATASET ----------------------- ", '\n')

# Number of steps to unroll
seq_dim = 60  # 180 90 60


def unique(list1):
    # initialize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    for x in unique_list:
        print(x, sep=' ')


'''
STEP 1: LOADING DATASET
'''


def loadData(files):
    print("Loading data... 📝")
    X = []
    print(f'File is: {files}')
    for i in files:
        df = pd.read_csv(f'C:\\read_thermal\SumAllCase\\{i}')
        df.drop('TimeStamp', axis=1, inplace=True)
        print(f'SUM Nan is {df.isna().sum().sum()}')
        df = df.dropna()
        X.append(df)
        print(f"{i} ✔")
    return pd.concat(X, axis=0, ignore_index=True)


def transformSequenceDataSplitLabel(df):
    X, Y = [], []
    video_frame = []
    data_num = 0

    print("\nTransforming data... 👨🏻‍💻")
    all_labels = df.iloc[:, -1].unique()
    print(f"All Labels: {all_labels}")

    for line in tqdm(range(len(df))):
        if data_num < seq_dim:
            frame2D = []
            frame = df.iloc[line, 0:-1]
            for h in range(height):
                frame2D.append([])
                for w in range(width):
                    t = frame[h * width + w]
                    frame2D[h].append(t)
            video_frame.append(frame2D)
            data_num += 1
        else:
            if (data_num == seq_dim):
                try:
                    Y.append(int(df.iloc[line, -1]))
                    X.append([video_frame])
                except:
                    pass
            video_frame = []
            data_num = 0
    try:
        # print(f"X:{X}")
        # print(f"Y:{Y}")
        X, Y = torch.FloatTensor(X), torch.LongTensor(Y)
        # X3 = torch.flatten(X, start_dim=3)
        # return X3, Y
        return X, Y
    except:
        pass


height = 8
width = 8

train_dataset = loadData(training_files)
print("train_dataset: ", train_dataset)
test_dataset = loadData(testing_files)
print("test_dataset: ", test_dataset)

X_train, y_train = transformSequenceDataSplitLabel(train_dataset)
X_test, y_test = transformSequenceDataSplitLabel(test_dataset)

print('Raw Training Images size: ', X_train.shape)
print('Raw Training Label size: ', y_train.shape,
      ' | Unique Label: ', y_train.unique(return_counts=True))
print('Raw Testing Images size: ', X_test.shape)
print('Raw Testing Label size: ', y_test.shape,
      ' | Unique Label: ', y_test.unique(return_counts=True))
# print("Train set shape:", X_train.shape, y_train.shape)
# print("Test set shape:", X_test.shape, y_test.shape)


class CustomTensorDataset(Dataset):

    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return self.data.size(0)


trainds = CustomTensorDataset(X_train, y_train)
testds = CustomTensorDataset(X_test, y_test)

print('\n', " --------------------- STEP 2: MAKING DATASET ITERABLE ----------------------- ", '\n')
'''
STEP 2: MAKING DATASET ITERABLE
'''

batch_size_list = [8, 16, 32, 64, 128, 256]
batch_size = batch_size_list[1]
num_epochs = 30

train_loader = DataLoader(dataset=trainds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=testds, batch_size=batch_size, shuffle=False)

train_features, train_labels = next(iter(train_loader))

print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

print(" --------------------- STEP 3: CREATE MODEL CLASS ----------------------- ", '\n')
'''
STEP 3: CREATE MODEL CLASS
'''


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        # self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        # self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.4)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        x = x.view(-1, x.shape[2], x.shape[3]*x.shape[4])

        h0 = torch.zeros(self.layer_dim, x.size(
            0), self.hidden_dim).requires_grad_().to(device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(
            0), self.hidden_dim).requires_grad_().to(device)

        # One time step
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out


print(" --------------------- STEP 4: INSTANTIATE MODEL CLASS ----------------------- ", '\n')
'''
STEP 4: INSTANTIATE MODEL CLASS
'''
input_dim = 64
hidden_dim = 100  # 200
layer_dim = 3  # 10
output_dim = 2

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)  # device = "cpu"
model = model.to(device)

# JUST PRINTING MODEL & PARAMETERS
print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

print(" --------------------- STEP 5: INSTANTIATE LOSS CLASS ----------------------- ", '\n')
'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.CrossEntropyLoss()
print(criterion)

print(" --------------------- STEP 6: INSTANTIATE OPTIMIZER CLASS ----------------------- ", '\n')
'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.001  # 0.01 0.001
momentum = 0.9
weight_decay = 0.001
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
print(optimizer)

'''
STEP 7: TRAIN THE MODEL
'''

print(" --------------------- STEP 7: TRAIN THE MODEL ----------------------- ", '\n')

train_score = []
test_score = []


train_loss = 0.0
train_total = 0
train_correct = 0
test_acc = 0.0
test_total = 0.0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_total = 0
    train_correct = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        model.train()

        # images = images.view(-1, seq_dim, input_dim).requires_grad_()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)  
        
        train_total += labels.size(0)

        train_correct += (predicted == labels).sum().item()

    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  #
            # images = images.view(-1, seq_dim, input_dim).requires_grad_()
            outputs = model(images)  # Forward pass only to get logits/output
            # Get predictions from the maximum value (0 or 1)
            _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # print("labels: ", labels)
            # print("predicted:", predicted)
            test_acc += (predicted == labels).sum().item()
            test_total += labels.size(0)


    if epoch % 1 == 0:
        
            TrainingLoss=train_loss / len(train_loader)

            Training_Accuracy=train_correct / train_total

            Testing_Accuracy=test_acc / test_total


    print(f'Epoch: {epoch+1}/{num_epochs} | Training Loss: {round(TrainingLoss,6)} | Training Accuracy: {round(Training_Accuracy,6)} | Testing Accuracy: {round(Testing_Accuracy,6)}')

    train_score.append(Training_Accuracy)
    test_score.append(Testing_Accuracy)






plt.grid(visible=True, which='major', axis='both',
         c='0.95', ls='-', linewidth=1.0, zorder=0)
plt.axhline(0.8, color="gold", linestyle="--",
            alpha=0.5, linewidth=1.0, label='base line')
plt.title(f'LSTM, learning_rate={learning_rate}, num_epochs={num_epochs}, hidden_dim={hidden_dim}, \n '
          f'layer_dim={layer_dim}, seq_dim={seq_dim}, batch_size={batch_size} optimizer={type(optimizer).__name__}')

plt.plot(range(1, num_epochs + 1), train_score, '--', label="Loss", color="g", alpha=0.5,
         linewidth=1.0)

plt.plot(range(1, num_epochs + 1),
         test_score, '--', label="Accuracy", color='b', alpha=0.5, linewidth=1.0)

plt.xticks(rotation=45, fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.xlabel('Training Epochs', fontsize=10)
plt.legend(fontsize=12, loc='upper right')

now=datetime.now()
dt_string=now.strftime("%d-%m-%Y_%H-%M-%S")
plt.savefig(f'./plots/{dt_string}.png')

plt.show()
