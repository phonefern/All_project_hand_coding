import math

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import cupy as cp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler
from sklearn.model_selection import KFold
import numpy as np
import torch.nn.functional as func
from tqdm import tqdm
import torch.utils.data as data_utils
import torch.nn.functional as F

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Step 1: Split your data
data_path = r"C:\read_thermal\SumAllCase\train_all_3_11.csv"
data = pd.read_csv(data_path)
print(data)
data.dropna(inplace=True)
data = data.drop(['TimeStamp'], axis=1)

# generating one row
data_sample = data.sample(frac=.001)

# display
print(data_sample)

X = data_sample.iloc[:, :-1].values
y = data_sample.iloc[:, -1].values

# Split the data into training (60%), validation (20%), and test (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

x_trainImages = []
x_valImages = []
x_testImages = []
y_trainLabels = []
y_valLabels = []
y_testLabels = []

height = 8
width = 8

def transformData(data, datatype):
    transformedData = []

    for ind in tqdm(range(len(data))):
        frame2D = []
        if datatype == "train":
            frame = X_train[ind]
        else:
            frame = X_test[ind]
        for h in range(height):
            frame2D.append([])
            for w in range(width):
                t = frame[h * width + w]
                frame2D[h].append(t)

        transformedData.append([frame2D])

    return transformedData

x_train_transformData = transformData(X_train, "train")
x_val_transformData = transformData(X_val, "val")
x_test_transformData = transformData(X_test, "test")

x_trainImages = torch.FloatTensor(x_train_transformData)
y_trainLabels = torch.LongTensor(y_train)
x_valImages = torch.FloatTensor(x_val_transformData)
y_valLabels = torch.LongTensor(y_val)
x_testImages = torch.FloatTensor(x_test_transformData)
y_testLabels = torch.LongTensor(y_test)

train_set = data_utils.TensorDataset(x_trainImages, y_trainLabels)
val_set = data_utils.TensorDataset(x_valImages, y_valLabels)
test_set = data_utils.TensorDataset(x_testImages, y_testLabels)

train_set = ConcatDataset([train_set, val_set])

# Step 2: Start with a simple model
class LeNet(nn.Module):
    def __init__(
            self,
            conv1_filters,
            conv1_kernel_size,
            conv2_filters,
            conv2_kernel_size,
            fc1_nodes,
            fc2_nodes,
            use_second_conv_block,
    ):
        super(LeNet, self).__init__()

        # input: (32, 1, 24, 24) batch * channel * width * height

        self.conv1 = nn.Conv2d(1, conv1_filters, conv1_kernel_size)
        # conv1 output = (32, 16, 22, 22) where 16 = conv1_filters, 22 = 24 - (conv1_kernel_size: 3) + 1
        self.pool = nn.MaxPool2d(2)
        # after pool2d pad (2,2): (32, 16, 11, 11)
        self.use_second_conv_block = use_second_conv_block

        if self.use_second_conv_block:
            self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, conv2_kernel_size)
            # conv2 output = (32, 32, 9, 9) where 9 = 11 - (conv2_kernel_size: 3) + 1
            # fc1 (input, fc1_nodes), where input = 32(conv2_filters) * h * w
            h = math.floor((((height - conv1_kernel_size + 1) / 2) - conv2_kernel_size + 1) / 2)
            w = math.floor((((width - conv1_kernel_size + 1) / 2) - conv2_kernel_size + 1) / 2)

            # conv2 output = (32, 32, 7, 7) where 7 = 9 - 3 + 1
            # fc1 (input, fc1_nodes), where input = 32(conv2_filters) * h * w
            #  h = 4, w = 4

            # if h > 1.6:
            #     h = math.floor(h)
            # else:
            #     h = math.ceil(h)
            # if w > 1.6:
            #     w = math.floor(w)
            # else:
            #     w = math.ceil(w)

            self.fc1 = nn.Linear(conv2_filters * h * w, fc1_nodes)

        else:
            # fc1 (input, fc1_nodes), where input = 16(conv1_filters) * ? * ?
            h = math.floor((height - conv1_kernel_size + 1) / 2)
            w = math.floor((width - conv1_kernel_size + 1) / 2)
            self.fc1 = nn.Linear(conv1_filters * h * w, fc1_nodes)
        self.fc2 = nn.Linear(fc1_nodes, fc2_nodes)
        self.fc3 = nn.Linear(fc2_nodes, 3)

    def forward(self, x):
        '''
        One forward pass through the network.

        Args:
            x: input
        '''
        # input: (32, 1, 24, 24) batch * channel * width * height

        x = self.pool(torch.relu(self.conv1(x)))
        # after convolution 1 (kernel= ex: 3, filters = 16): (32, 16, 22, 22) where 22=24-3+1
        # after pool2d pad (2,2): (32, 16, 11, 11)

        # after convolution 1 (kernel= ex: 7, filters = 16): (32, 16, 18, 18) where 18=24-7+1
        # after pool2d pad (2,2): (32, 16, 9, 9)

        if self.use_second_conv_block:
            x = self.pool(torch.relu(self.conv2(x)))
            # after convolution 2 (kernel= ex: 3, filters: 32): (32, 32, 9, 9) where 9=11-3+1
            # after pool2d pad (2,2): (32, 32, 4, 4)

            # after convolution 2 (kernel= ex: 7, filters: 32): (32, 32, 7, 7) where 7=9-3+1
            # after pool2d pad (2,2): (32, 32, 3, 3) -> 32 * 3 * 3 -> (32, 288)

        x = x.view(-1, self.num_flat_features(x))
        # if use conv2: x.view, transform data to 1D tensor -> input (32, 32, 4, 4) -> (32, 32 * 4 * 4) -> (32, 512)
        # if does not use conv2: x.view, transform data to 1D tensor -> input (32, 16, 11, 11) -> (32, 16 * 11 * 11) -> (32, 1936)
        # this output size have to match with fc1 input, otherwise it throws an error (mat1 and mat2 shapes cannot be multiplied)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)


# Step 3: Define a search space
params = {
    'batch_size': [32, 64, 128],
    'num_epochs': [40, 50, 60, 100],
    'lr': [0.01, 0.001, 0.0001],
    "weight_decay": [0.01, 0.001, 0.0001],
    'conv1_filters': [8, 16, 32],
    'conv1_kernel_size': [2, 3],
    'conv2_filters': [16, 32, 64],
    'conv2_kernel_size': [2, 3],
    'fc1_nodes': [64, 128, 256],
    'fc2_nodes': [32,64, 128],
    'use_second_conv_block': [True, False]
}

# Step 5: Perform cross-validation
# Define number of folds
k = 5
# Split data into k folds
kfold = KFold(n_splits=k, shuffle=True, random_state=42)

# Step 6: Select the best model
# Perform cross-validation to estimate performance for each parameter setting
best_perf = -float('inf')
best_params = None

lr_range = cp.random.choice(params["lr"], len(params["lr"]))
num_epochs_range = cp.random.choice(params["num_epochs"], len(params["num_epochs"]))
weight_decay_range = cp.random.choice(params["weight_decay"], len(params["weight_decay"]))
batch_size_range = cp.random.choice(params["batch_size"], len(params["batch_size"]))
conv1_filters_range = cp.random.choice(params["conv1_filters"], len(params["conv1_filters"]))
conv1_kernel_size_range = cp.random.choice(params["conv1_kernel_size"], len(params["conv1_kernel_size"]))
conv2_filters_range = cp.random.choice(params["conv2_filters"], len(params["conv2_filters"]))
conv2_kernel_size_range = cp.random.choice(params["conv2_kernel_size"], len(params["conv2_kernel_size"]))
fc1_nodes_range = cp.random.choice(params["fc1_nodes"], len(params["fc1_nodes"]))
fc2_nodes_range = cp.random.choice(params["fc2_nodes"], len(params["fc2_nodes"]))
use_second_conv_block_range = cp.random.choice(params["use_second_conv_block"], len(params["use_second_conv_block"]))

for num_epochs in num_epochs_range:
    num_epochs = int(num_epochs)
    for lr in lr_range:
        lr = float(lr)
        for weight_decay in weight_decay_range:
            weight_decay = float(weight_decay)
            for batch_size in batch_size_range:
                print()
                print("*************************************************")
                print(" num_epochs: {}, lr: {}, weight_decay: {}, batch_size: {}".format(num_epochs, lr, weight_decay,
                                                                                        batch_size))

                batch_size = int(batch_size)
                print("*************************************************")

                dataset = ConcatDataset([train_set, test_set])

                for conv1_filters in conv1_filters_range:
                    for conv1_kernel_size in conv1_kernel_size_range:
                        for conv2_filters in conv2_filters_range:
                            for conv2_kernel_size in conv2_kernel_size_range:
                                for fc1_nodes in fc1_nodes_range:
                                    for fc2_nodes in fc2_nodes_range:
                                        for use_second_conv_block in use_second_conv_block_range:

                                            conv1_filters = int(conv1_filters)
                                            conv1_kernel_size = int(conv1_kernel_size)
                                            conv2_filters = int(conv2_filters)
                                            conv2_kernel_size = int(conv2_kernel_size)
                                            fc1_nodes = int(fc1_nodes)
                                            fc2_nodes = int(fc2_nodes)
                                            use_second_conv_block = bool(use_second_conv_block)

                                            perf_sum = 0

                                            print("   conv1_filters: {}, conv1_kernel_size: {}, conv2_filters: {}, "
                                                  "conv2_kernel_size: {}, fc1_nodes: {}, fc2_nodes: {}, "
                                                  "use_second_conv_block: {}".format(conv1_filters, conv1_kernel_size,
                                                                                     conv2_filters, conv2_kernel_size,
                                                                                     fc1_nodes, fc2_nodes, use_second_conv_block))

                                            for fold_idx, (train_idx, val_idx) in enumerate(
                                                    kfold.split(np.arange(len(dataset)))):

                                                train_sampler = SubsetRandomSampler(train_idx)
                                                val_sampler = SubsetRandomSampler(val_idx)

                                                # # Create data loaders for training and validation folds
                                                train_loader = DataLoader(dataset, batch_size=batch_size,
                                                                          sampler=train_sampler)
                                                val_loader = DataLoader(dataset, batch_size=batch_size,
                                                                        sampler=val_sampler)

                                                # Initialize model and optimizer
                                                model = LeNet(conv1_filters, conv1_kernel_size, conv2_filters,
                                                              conv2_kernel_size, fc1_nodes, fc2_nodes,
                                                              use_second_conv_block).to(device)
                                                optimizer = optim.Adam(model.parameters(), lr=lr,
                                                                       weight_decay=weight_decay)

                                                # Train model on training fold and evaluate on validation fold
                                                for epoch in range(num_epochs):
                                                    print(".", end="")
                                                    model.train()
                                                    for batch_idx, (data, target) in enumerate(train_loader):
                                                        data, target = data.to(device), target.to(device)
                                                        optimizer.zero_grad()
                                                        output = model(data)
                                                        loss = nn.CrossEntropyLoss()(output, target)
                                                        loss.backward()
                                                        optimizer.step()

                                                    model.eval()
                                                    val_loss = 0
                                                    correct = 0
                                                    with torch.no_grad():
                                                        for data, target in val_loader:
                                                            data, target = data.to(device), target.to(device)
                                                            output = model(data)
                                                            val_loss += nn.CrossEntropyLoss()(output, target).item()
                                                            pred = output.max(1, keepdim=True)[1]
                                                            correct += pred.eq(target.view_as(pred)).sum().item()
                                                        val_loss /= len(val_loader.dataset)
                                                        accuracy = 100. * correct / len(val_loader.dataset)
                                                    perf_sum += accuracy

                                                print()

                                            # Calculate average performance across all folds
                                            perf_avg = perf_sum / k

                                            # Check if this parameter setting is better than the current best
                                            if perf_avg > best_perf:
                                                best_perf = perf_avg
                                                best_params = {'lr': lr, 'weight_decay': weight_decay,
                                                               'batch_size': batch_size,
                                                               'conv1_filters': conv1_filters,
                                                               'conv1_kernel_size': conv1_kernel_size,
                                                               'conv2_filters': conv2_filters,
                                                               'conv2_kernel_size': conv2_kernel_size,
                                                               'fc1_nodes': fc1_nodes, 'fc2_nodes': fc2_nodes,
                                                               'use_second_conv_block': use_second_conv_block}
                                                print("******** New best parameters found: ", best_params)
                                                print()

print()
print()
print("******** Train the final model on the full training set with the best hyperparameters.")
print("******** Best parameters found: ", best_params)
print()

# Step 7: Train the final model and evaluate on the test set
# Train the final model on the full training set with the best hyperparameters
model = LeNet(best_params['conv1_filters'], best_params['conv1_kernel_size'], best_params['conv2_filters'],
              best_params['conv2_kernel_size'], best_params['fc1_nodes'], best_params['fc2_nodes'],
              best_params['use_second_conv_block']).to(device)
train_loader = DataLoader(train_set, batch_size=best_params['batch_size'], shuffle=True)
test_loader = DataLoader(test_set, batch_size=best_params['batch_size'], shuffle=False)

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        print(".", end="")
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

print()
print("******** Evaluate final model on test set.")
print()

# Evaluate final model on test set
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += nn.CrossEntropyLoss()(output, target).item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
test_loss /= len(test_loader.dataset)
accuracy = 100. * correct / len(test_loader.dataset)

print()
print("Test set accuracy: {:.2f}%".format(accuracy))
