import os
from datetime import datetime, timedelta
import time
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
import matplotlib.pylab as plt
import torch.nn.functional as F
import gc
from torch import nn
from pygame import mixer
from colorama import Fore, Back, Style
from statistics import mean
import time
import inspect
from tqdm import tqdm


def get_time_hh_mm_ss(sec):
    td_str = str(timedelta(seconds=sec))
    # split string into individual component
    x = td_str.split(':')
    x_str = f'hh:mm:ss: {x[0]} Hours {x[1]} Minutes {x[2]} Seconds'
    return x_str

def Average(lst):
    return sum(lst) / len(lst)

gc.collect()
torch.cuda.empty_cache()

layer_cnn = 2  # !!! 4
dropout = "nodrop"  # !!!

torch.backends.cudnn.enabled = False

# training_files = ['train_all_3_11.csv']
# testing_files = ['train_all_1_11_split.csv']

training_files = ['Balance_01.csv']
testing_files = ['split_50k_2.csv']

X_train = []
X_test = []
y_train = []
y_test = []

### Preprocessing
print('\n ---- Preprocessing Starting ---- ', '\n')

timesteps = 180 #เป็นตัวกำหนดว่าต้องได้180ภาพก่อนนะค่อยกลับมาดู0ใหม่
print(f'timesteps: {timesteps}')
num_classes = 2
print(f'num_classes: {num_classes}')

width = 8
height = 8
print(f'width {width} * height {height}')

lines = '\n-----------------------------------------------------------------------\n'


def loadData(files):
    X = []

    for i in files:
        df = df = pd.read_csv(f'C:\\read_thermal\SumAllCase\\{i}')
        df.drop('TimeStamp', axis=1, inplace=True)
        df = df.dropna()
        X.append(df)
    return pd.concat(X, axis=0, ignore_index=True)


def transformSequenceData(df):
    X, Y = [], []
# video_frameเป็นการกำหนดตัวแปรวกำหนดชุดภาพที่วนครบ180
    video_frame = []
    data_num = 0

    len_label_0 = len(df[df['Label'] == 0])
    len_label_1 = len(df[df['Label'] == 1])
    print(f'Length Lable 0: {len_label_0}')
    print(f'Length Label 1: {len_label_1}')
#บรรดทันพวกนี้จะเป็นการกำหนดLabelว่า0,1

    for line in tqdm(range(len(df))):
        if data_num < timesteps: #ตัวtimestepsตรงนี้แค่ดึงค่าตามที่ตกลงว่าให้ครบ 180 ภาพก่อนแล้ววนกลับมาดู
            frame2D = []
            frame = df.iloc[line, 0:-1]
            for h in range(height):
                frame2D.append([])
                for w in range(width): #โดยจะดึงค่าwidthที่กำหนดไว้มาใช้
                    t = frame[h * width + w]
                    frame2D[h].append(t)

            video_frame.append([frame2D])
            data_num = data_num + 1
        else:
            Y.append(df.iloc[line, -1])
            X.append(video_frame)
            video_frame = []
            data_num = 0
#จากบรรดทัดแรกจนถึงบรรดทัดนี้ จะเป็นการเริ่มสร้างภาพเท่านั้นยังได้กะเถิบไปข้างในเลย
    #ปิดครอสวันที่15/11/2566
    X, Y = torch.FloatTensor(X), torch.LongTensor(Y)

    return X, Y


print('\n---- Reading files ----\n')

X_train = loadData(training_files)
lengthXtrain = len(X_train)
print('Length X_train', lengthXtrain)
X_test = loadData(testing_files)
lengthXtest = len(X_test)
print('Length X_test', lengthXtest)

print('\n---- Transforming Sequence Data ----\n')

X_train, y_train = transformSequenceData(X_train)
X_test, y_test = transformSequenceData(X_test)

print('Raw Training Images size: ', X_train.shape)
print('Raw Training Label size: ', y_train.shape, ' | Unique Label: ', y_train.unique(return_counts=True))
print('Raw Testing Images size: ', X_test.shape)
print('Raw Testing Label size: ', y_test.shape, ' | Unique Label: ', y_test.unique(return_counts=True))

print('\n', "----------Defining Data Loaders----------", '\n')

batch_size = 64
layer_cnn = 2  # [2,4,8,16,32,64,128,256] *Out of memory [512, 1024]
print(f'batch_size: {batch_size}')

# get the training set
train_set = TensorDataset(X_train, y_train)  # สร้าง datasets สำหรับ train
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)  # สร้าง dataloader สำหรับ train set

# get the test set
test_set = TensorDataset(X_test, y_test)  # สร้าง datasets สำหรับ test
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)  # สร้าง dataloader สำหรับ test set

print('\n', "----------Defining Model----------", '\n')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Conv Layer 1

        # Input size: 8*8*1
        # Spatial extend of each one (kernelConv size), F = 2
        # Slide size (strideConv), S = 1
        # Padding, P = 0
        ## Width: ((8 - 2 + 2 * 0) / 1) + 1 = 7.0
        ## High: ((8 - 2 + 2 * 0) / 1) + 1 = 7.0
        ## Depth: 16
        ## Output Conv Layer1: 7.0 * 7.0 * 16

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, stride=1, padding=0)
        self.batch1 = torch.nn.BatchNorm2d(16)  # **optional
        self.drop1 = torch.nn.Dropout2d(0.1)  # **optional
        self.relu1 = torch.nn.ReLU()

        # Max Pooling Layer 1
        # Input size: 7.0 * 7.0 * 16
        ## Spatial extend of each one (kernelMaxPool size), F = 2
        ## Slide size (strideMaxPool), S = 1

        # Output Max Pooling Layer 1
        ## Width: ((7.0 - 2) / 1) + 1 = 6.0
        ## High: ((7.0 - 2) / 1) + 1 = 6.0
        ## Depth: 16
        ### Output Max Pooling Layer 1: 6.0 * 6.0 * 16

        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=1)

        # Conv Layer 2

        # Input size: 6.0*6.0*16
        # Spatial extend of each one (kernelConv size), F = 2
        # Slide size (strideConv), S = 1
        # Padding, P = 0
        ## Width: ((6.0 - 2 + 2 * 0) / 1) + 1 = 5.0
        ## High: ((6.0 - 2 + 2 * 0) / 1) + 1 = 5.0
        ## Depth: 32
        ## Output Conv Layer2: 5.0 * 5.0 * 32

        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=0)
        self.batch2 = torch.nn.BatchNorm2d(32)  # **optional
        self.drop2 = torch.nn.Dropout2d(0.1)  # **optional
        self.relu2 = torch.nn.ReLU()

        # Max Pooling Layer 2
        # Input size: 5.0 * 5.0 * 32
        ## Spatial extend of each one (kernelMaxPool size), F = 2
        ## Slide size (strideMaxPool), S = 1

        # Output Max Pooling Layer 2
        ## Width: ((5.0 - 2) / 1) + 1 = 4.0
        ## High: ((5.0 - 2) / 1) + 1 = 4.0
        ## Depth: 32
        ### Output Max Pooling Layer 2: 4.0 * 4.0 * 32

        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=1)

    def forward(self, i):
        x = i.view(-1, i.shape[2], i.shape[3], i.shape[4])

        # Conv 1
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.relu1(out)

        # Max Pool 1
        out = self.pool1(out)

        # Conv 2
        out = self.conv2(out)
        out = self.batch2(out)
        out = self.relu2(out)

        # Max Pool 2
        out = self.pool2(out)

        out = out.view(i.shape[0], i.shape[1], -1)
        # print(out.shape)
        return out


# Customize the model LSTM

input_size = 4 * 4 * 32 # for 2 layers CNN
hidden_size = 100  # Hidden Unit
layer_size = 2  # Hidden Layers


class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, hidden_size, layer_size, batch_first=True)

        # Output Size = Number of classes = 3
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # print(x.shape)

        # x, _ = self.lstm(x)
        # x = x.view(x.shape[0], -1)
        # x = self.fc(x)
        # return x

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_size, x.size(0), self.hidden_size).requires_grad_().to(device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_size, x.size(0), self.hidden_size).requires_grad_().to(device)

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)

# Test loss and accuracy
training_loss = []
train_score = []
test_score = []

# Customize the training process
num_epochs = 5  # [50, 100, 150, 250, 300]
# num_epochs = []
learning_rate = 0.001
weight_decay = 0.001

criterion = torch.nn.CrossEntropyLoss()

print('Start training...')

time_training = []

now=datetime.now()
dt_string=now.strftime("%d-%m-%Y_%H-%M-%S")

rounds = 1

for i in range(rounds):
    file_name = f'./model_save/CNN-LSTM--Batch_size-{batch_size}--Learning_rate-{learning_rate}--Layer_size_LSTM--{layer_size}-Layer_size_cnn-{layer_cnn}-Epochs--{num_epochs}--Dropout(0.1)--round--{rounds}--{dt_string}'
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    print(f'File name: {file_name}')
    with open(os.path.join(file_name, 'training_details.txt'), 'w') as file:
        now = datetime.now()
        dt_string = now.strftime("%d-%m_%H-%M-%S")
        print(f'Start {rounds} at {dt_string}')

        start_time_round = time.time()
        epochList = []
        lossList = []
        trainAccList = []
        testAccList = []

        print(lines)

        loss_num = 0

        # Model Setting
        cnn_model = CNN().to(device)
        cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # cnn_optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate, momentum=0.9)
        lstm_model = LSTM().to(device)
        lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        #lstm_optimizer = torch.optim.SGD(lstm_model.parameters(), lr=learning_rate, momentum=0.9)

        # for name, param in cnn_model.named_parameters():
        #     if param.requires_grad:
        #         print("model_cnn: ", name, param.shape)
        #
        # for name, param in lstm_model.named_parameters():
        #     if param.requires_grad:
        #         print("model_lstm: ", name, param.shape)

        for epoch in range(num_epochs):
            start_time = time.time()
            epochList.append(epoch)
            train_loss = 0.0
            train_total = 0
            train_correct = 0
            times = []

            for batch_idx, (data, target) in enumerate(train_loader):
                # print("batch_idx: ", batch_idx)
                # print("data.shape: ", data.shape)
                # print("target.shape: ", target.shape)

                data, target = data.to(device), target.to(device)

                # Clear gradients
                cnn_optimizer.zero_grad()
                lstm_optimizer.zero_grad()

                cnn_model.train()
                lstm_model.train()
                features = cnn_model(data)
                outputs = lstm_model(features)

                loss = criterion(outputs, target)
                loss.backward()

                # Update parameters
                cnn_optimizer.step()
                lstm_optimizer.step()
                train_loss += loss.item()
                loss_num += 1

                _, predicted = torch.max(outputs.data, 1)

                # print("predicted_train: ", predicted, "target: ", target)

                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()

            test_acc = 0.0
            test_total = 0.0
            cnn_model.eval()
            lstm_model.eval()

            with torch.no_grad():
                for batch_idx, (inputs, y_true) in enumerate(test_loader):
                    # Extracting images and target labels for the batch being iterated
                    inputs, y_true = inputs.to(device), y_true.to(device)

                    # Calculating the model output and the cross entropy loss
                    features = cnn_model(inputs)
                    outputs = lstm_model(features)

                    # Calculating the accuracy of the model
                    _, predicted_test = torch.max(outputs.data, 1)

                    # print("\n predicted_test: ", predicted_test, "y_true: ", y_true, "\n")

                    test_acc += (predicted_test == y_true).sum().item()  # นับ accuracy
                    test_total += y_true.size(0)

            endtime = time.time()
            usetime = endtime - start_time
            times.append(usetime)

            trainAccList.append(train_correct / train_total)
            testAccList.append(test_acc / test_total)
            if (train_loss / loss_num) > 1.0:
                lossList.append(1)
            else:
                lossList.append(train_loss / loss_num)

            # คำนวณค่า accuracy และ loss ของ train set และ test set
            if epoch % 2 == 0:
                TrainingLoss = train_loss / loss_num
                Training_Accuracy = train_correct / train_total
                Testing_Accuracy = test_acc / test_total
                TimeMean = mean(times)

                detail = (
                    f'Round {i + 1} \tEpoch: {epoch + 1}/{num_epochs} \tTraining Loss: {TrainingLoss:.10f} \tTraining Accuracy: {Training_Accuracy:.10f} \tTesting Accuracy: {Testing_Accuracy:.10f} \tTime Learning: {TimeMean:.3f} Sec. BatchSize {batch_size}\n')
                print(detail)
                file.write(detail)
                time_training.append(TimeMean)

            if epoch == num_epochs - 1:
                training_loss.append(TrainingLoss)

        end_time_round = time.time()
        time_per_round = end_time_round - start_time_round
        time_left = time_per_round * (rounds - i - 1)
        print(
            f'Time round {i + 1}: {get_time_hh_mm_ss(time_per_round)} Sec. About time left {get_time_hh_mm_ss(round(time_left))}.')

file.close()
time.sleep(5)

plt.grid(visible=True, which='major', axis='both', c='0.95', ls='-', linewidth=1.0, zorder=0)
plt.title(
    f"CNN-LSTM,learning_rate={learning_rate},batch_size={batch_size},epoch={num_epochs},\nweight_decay={weight_decay},timesteps={timesteps},cnnOptim={type(cnn_optimizer).__name__},lstmOptim={type(lstm_optimizer).__name__}")
plt.plot(lossList, label='Loss', color="blue", alpha=0.5, linewidth=1.0)
plt.plot(trainAccList, label='Train', color="green", alpha=0.5, linewidth=1.0)
plt.plot(testAccList, label='Test', color="darkorange", alpha=0.5, linewidth=1.0)
plt.xticks(rotation=45, fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.xlabel('Times', fontsize=10)
plt.legend(fontsize=8, loc='lower right')


plt.savefig(os.path.join(file_name, 'CNN_plot.png'), dpi=300)
plt.show()
print("Save plot success")


# Save the trained model and optimizer

# torch.save(outputs.state_dict(), f'./model/cnn_lstm.pth')
# torch.save({'cnn_model': cnn_model.state_dict(), 'lstm_model': lstm_model.state_dict(), 'cnn_optimizer': cnn_optimizer.state_dict(), 'lstm_optimizer': lstm_optimizer.state_dict()}, 'model_and_optimizers.pth')

    
