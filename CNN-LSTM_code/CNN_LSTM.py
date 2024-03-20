import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101
import numpy
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pylab as plt
from tqdm import tqdm
import os
from alive_progress import alive_bar
from time import sleep
import icecream as ic


# torch.backends.cudnn.enabled = False
torch.cuda.empty_cache()

training_files = ['train.csv']
testing_files = ['test.csv']

print(" --------------------- STEP 1: LOADING DATASET ----------------------- ", '\n')

# Number of steps to unroll
seq_dim = 60

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
    for i in files:
        df = pd.read_csv('C:\\read_thermal\SumAllCase\\Train_all.csv')
        df.drop('TimeStamp', axis=1, inplace=True)
        # if(i=='nohuman.csv'):df.drop(df.index[0:90000],inplace=True)
        # if(i=='nohuman.csv'):df.drop(df.index[0:90000],inplace=True)
        X.append(df)
        # print(df.head(2))
        print(f"{i} ✔")
        print(df.iloc[:, -1].unique())
    # print(X)
    return pd.concat(X, axis=0, ignore_index=True)

def transformSequenceDataSplitLabel(df):
    X, Y = [], []
    video_frame = []
    data_num = 0
    print("\nTransforming data... 👨🏻‍💻")
    all_labels = df.iloc[:, -1].unique()
    print(f"All Labels: {all_labels}")
    for l in all_labels:
        print(f"Selected Label: {l}")
        df_lable = df.loc[df['Label'] == l]

        print(f"Length Label {l} : {len(df_lable)}")
        print(f"Length Label {l} / 180 : {len(df_lable) / 180}")

        for line in range(len(df_lable)):
            if data_num < seq_dim:
                # print(f"Data Num: {data_num}")
                frame2D = []
                frame = df_lable.iloc[line, 0:-1]
                for h in range(height):
                    frame2D.append([])
                    for w in range(width):
                        t = frame[h * width + w]
                        frame2D[h].append(t)
                video_frame.append(frame2D)
                data_num += 1
            else:
                if (data_num == seq_dim):
                    # if df.iloc[line, -1] == 2:
                    #    Y.append(1)
                    #elif df.iloc[line, -1] == 3:
                    #    Y.append(2)
                    #else:
                    #    Y.append(0)
                    label_now = df_lable[line - 60:line].iloc[:, -1].unique()
                    # print(f'\nLabel Now : {label_now}')
                    Y.append(df_lable.iloc[line, -1])
                    X.append([video_frame])
                video_frame = []
                data_num = 0
        print(f"Label:{l} ✔")
    # unique(Y)
    X, Y = torch.FloatTensor(X), torch.LongTensor(Y)
    # print("X: ", X[0])
    X3 = torch.flatten(X, start_dim=3)
    print("X3-flatten: ", X3.shape)
    return X3, Y

height = 8
width = 8



train_dataset = loadData(training_files)
test_dataset = loadData(testing_files)

X_train, y_train = transformSequenceDataSplitLabel(train_dataset)
X_test, y_test = transformSequenceDataSplitLabel(test_dataset)

# Remove rows with NaN in the Label column
X_train = [X_train[i] for i in range(len(y_train)) if not np.isnan(y_train[i])]
y_train = [y_train[i] for i in range(len(y_train)) if not np.isnan(y_train[i])]

X_test = [X_test[i] for i in range(len(y_test)) if not np.isnan(y_test[i])]
y_test = [y_test[i] for i in range(len(y_test)) if not np.isnan(y_test[i])]



print("Train set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

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

batch_size = 100
n_iters = 10
num_epochs = 10

train_loader = DataLoader(dataset=trainds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=testds, batch_size=batch_size, shuffle=False)

train_features, train_labels = next(iter(train_loader))

print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

print(" --------------------- STEP 3: CREATE MODEL CLASS ----------------------- ", '\n')
'''
STEP 3: CREATE MODEL CLASS
'''

class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(CNNLSTM, self).__init__()
        self.resnet = resnet101(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, input_dim))
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, output_dim)
       
    def forward(self, x_3d):
        print("x_3d: ", x_3d.shape)
        print("x_3d: ", x_3d)
        print("x_3d: ", x_3d.size(1))
        hidden = None
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :])
            out, hidden = self.lstm(x.unsqueeze(0), hidden)         

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x

print(" --------------------- STEP 4: INSTANTIATE MODEL CLASS ----------------------- ", '\n')
'''
STEP 4: INSTANTIATE MODEL CLASS
'''

input_dim = 64
hidden_dim = 100
layer_dim = 10  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 3

model = CNNLSTM(input_dim, hidden_dim, layer_dim, output_dim)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(optimizer)

'''
STEP 7: TRAIN THE MODEL
'''

print(" --------------------- STEP 7: TRAIN THE MODEL ----------------------- ", '\n')

loss_history = {
    "train": [],
    "accuracy": [],
}

iter = 0

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        model.train()

        # print(labels.min(), labels.max())

        # print(f'Epoch: {epoch + 1}/{num_epochs} | Step: {i + 1}/{len(train_loader)} | Batch size: {images.size(0)}')

        images = images.to(device)
        labels = labels.to(device)

        # print("enumerate-images: ", images.shape)
        # print("enumerate-labels: ", labels.shape)

        # Load images as tensors with gradient accumulation abilities
        images = images.view(-1, seq_dim, input_dim).requires_grad_()

        # print("images.view: ", images.shape)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        # outputs.size() --> 100, 10
        outputs = model(images)

        # print("outputs: ", outputs.shape, outputs)
        # print("labels: ", labels.shape, labels)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        # print((num_epochs*len(train_loader))//10)

        if iter % ((num_epochs * len(train_loader)) // 20) == 0:
            # if iter % 2 == 0:

            # print(" --------------------- ITER ----------------------- ", '\n')

            model.eval()

            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                # Resize images
                images = images.view(-1, seq_dim, input_dim)

                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted.cpu() == labels.cpu()).sum()

                print("predicted: ", predicted.cpu().numpy())
                print("labels: ", labels.cpu())
                # print("Predicted == Labels :", (predicted.cpu() == labels.cpu()).sum())

            # accuracy = 100 * correct / total

            accuracy = correct / total

            loss_history["train"].append(loss.item())
            loss_history["accuracy"].append(accuracy.item())

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))

            # print(" --------------------- ITER ----------------------- ", '\n')

num_epochs = len(loss_history["train"])

plt.grid(visible=True, which='major', axis='both', c='0.95', ls='-', linewidth=1.0, zorder=0)
plt.axhline(0.8, color="gold", linestyle="--", alpha=0.5, linewidth=1.0, label='base line')
plt.title(f'CNN-LSTM, learning_rate={learning_rate}, num_epochs={num_epochs}, hidden_dim={hidden_dim}, \n '
          f'layer_dim={layer_dim}, seq_dim={seq_dim}, batch_size={batch_size} optimizer={type(optimizer).__name__}')
plt.plot(range(1, num_epochs + 1), loss_history["train"], label="Loss", color="g", alpha=0.5,
         linewidth=1.0)
plt.plot(range(1, num_epochs + 1), loss_history["accuracy"], label="Accuracy", color='b', alpha=0.5, linewidth=1.0)
plt.xticks(rotation=45, fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.xlabel('Training Epochs', fontsize=10)
plt.legend(fontsize=12, loc='upper right')

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
plt.savefig(f'./plots/{dt_string}.png')

plt.show()
