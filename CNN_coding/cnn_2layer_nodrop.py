import pandas as pd
import torch
import torchvision
import numpy as np
from sklearn.model_selection import train_test_split
import ssl
import csv
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import time
from statistics import mean
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report, \
    ConfusionMatrixDisplay

line = '\n-----------------------------------------------------------------------'

def get_time_hh_mm_ss(sec):
    td_str = str(timedelta(seconds=sec))
    # split string into individual component
    x = td_str.split(':')
    x_str = f'hh:mm:ss: {x[0]} Hours {x[1]} Minutes {x[2]} Seconds'
    return x_str

def Average(lst):
    return sum(lst) / len(lst)

# cnn class
class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            
            # ข้อมูลขนาด 8x8 ผ่าน kenel 2x2 เเละ padding 1 = 9x9   เเละเมื่อผ่านconvolution จะได้ขนาด 8x8x8 เท่าเดิม ผ่าน relu จะได้ 8x8x8 ผ่าน pooling จะได้ 8x8 -> 4x4 output 8x4x4
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3), padding=0),  # คอนโว = layerเเรก
            # torch.nn.BatchNorm2d(8), # ปรับค่าให้มี mean = 0 และ std = 1 ในแต่ละชั้น
            torch.nn.Dropout(0.4), # ลด overfitting โดยการปิดบางเเถว ในการเทรน
            torch.nn.ReLU(), # Rectified Linear Unit ถ้าค่าเป็นบวกให้เอาค่านั้นไป ถ้าเป็นลบให้เอาค่า 0 ไป
            torch.nn.MaxPool2d(kernel_size=2, stride=1), # เลือกค่าที่มากที่สุดใน 2x2 จะได้ 4x4


            
            torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(2), padding=0),  #เหลือ 2x2 output 16x2x2
            # torch.nn.BatchNorm2d(16),
            torch.nn.Dropout(0.4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=1),

            torch.nn.Flatten(),
            torch.nn.Linear(16 * 3 * 3, 2),  # ชั้น Fully Connected Layer หลังจาก Linear: 4 -> 32
            # torch.nn.ReLU(),
            # torch.nn.Linear(288, 4) #  หลังจาก Linear: 288 -> 4
        )

    def forward(self, x):
        return self.model(x)


# list for the training set
x_train = []
x_test = []

y_train = []
y_test = []

lines = '\n-----------------------------------------------------------------------\n'

# # setting dataset for test && train
# testing_files = ['testing-merge']
# training_files = ['training-merge']
# print(lines)
#
# # reading the csv files testings set
# print('Loading data...')
# # for i in testing_files:
# #     # df = pd.read_csv('C:\\Users\\Admin\\Desktop\\test-merge\\' + str(i) + '.csv', encoding="utf8",
# #     #                  on_bad_lines='warn')
# df = pd.read_csv('D:\\project_wu_smart_building\\test-merge.csv', encoding="utf8",
#                      on_bad_lines='warn')
# for row in range(len(df)):
#     y_test.append(df.iloc[row, -1])
#     x_test.append(df.iloc[row, 1:-1])
#
# # reading the csv files trainings set
# # for j in training_files:
#     # df = pd.read_csv('C:\\Users\\Admin\\Desktop\\train-merge\\' + str(j) + '.csv', encoding="utf8",
#     #                  on_bad_lines='warn')
# df = pd.read_csv('D:\\project_wu_smart_building\\train-merge.csv', encoding="utf8",
#                      on_bad_lines='warn')
# for row in range(len(df)):
#     y_train.append(df.iloc[row, -1])
#     x_train.append(df.iloc[row, 1:-1])

# Load your data and preprocess it
df = pd.read_csv(r"C:\read_thermal\SumAllCase\Train_data.csv")
x = df.iloc[:, 1:65].values
y = df['Label'].values
# Data Splitting for training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.50)

len_label_0 = len(df[df['Label'] == 0])
len_label_1 = len(df[df['Label'] == 1])
print(f'Length Lable 0: {len_label_0}')
print(f'Length Label 1: {len_label_1}')

# Remove rows with NaN in the Label column
x_train = [x_train[i] for i in range(len(y_train)) if not np.isnan(y_train[i])]
y_train = [y_train[i] for i in range(len(y_train)) if not np.isnan(y_train[i])]

x_test = [x_test[i] for i in range(len(y_test)) if not np.isnan(y_test[i])]
y_test = [y_test[i] for i in range(len(y_test)) if not np.isnan(y_test[i])]

print('Raw Training Images size: ', len(x_train))
print('Raw Training Label size: ', len(y_train), ' | Unique Label: ', np.unique(np.array(y_train)))
print('Raw Testing Images size: ', len(x_test))
print('Raw Testing Label size: ', len(y_test), ' | Unique Label: ', np.unique(np.array(y_test)))

print(lines)
print('Data preprocessing...')

# แปลงรูปของดาต้าให้อยู่ในรูปแบบของ tensor
x_trainImages = []
x_testImages = []
y_trainLabels = []
y_testLabels = []

Rawdata_train = [x_train, x_test]
                                        
data = [x_trainImages, x_testImages]

num = 0
for d in Rawdata_train:
    # Data Transforming
    for i in d:
        frame2D = []
        for h in range(8):
            frame2D.append([])
            for w in range(8):
                t = i[h * 8 + w]
                frame2D[h].append(t)

        data[num].append([frame2D])

    num += 1

x_trainImages = torch.FloatTensor(x_trainImages)
y_trainLabels = torch.LongTensor(y_train)
x_testImages = torch.FloatTensor(x_testImages)
y_testLabels = torch.LongTensor(y_test)

# Data Loader
print('Transformed X_trainImages Images size: ', x_trainImages.size())
print('Transformed Y_trainLabels Labels size: ', y_trainLabels.size())
print('Transformed X_testImages Images size: ', x_testImages.size())
print('Transformed Y_testLabels Labels size: ', y_testLabels.size())

# ปรับ device ให้เป็น GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(lines)
print('Using {} device'.format(device))
model = CNN().to(device)

# Hyperparameters for training the model (you can change it)

num_epochs = 50 # จำนวนรอบในการเทรน
learning_rate = 0.001 # ความเร็วในการเทรน
weight_decay = 0.001 # ค่าความเร็วในการเทรน
batch_size = 32
layer_size = 2 # 
criterion = torch.nn.CrossEntropyLoss() # คำนวณค่า loss ของโมเดล

# Model Setting
model = CNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) #  Adaptive Moment Estimation ปรับ lerning late ให้เหมาะสมกับแต่ละรอบ ลดการเกิด overfitting

# get the training set
train_set = data_utils.TensorDataset(x_trainImages, y_trainLabels)  # สร้าง datasets สำหรับ train
train_loader = data_utils.DataLoader(train_set, batch_size=batch_size, shuffle=True)  # สร้าง dataloader สำหรับ train set

# get the test set
test_set = data_utils.TensorDataset(x_testImages, y_testLabels)  # สร้าง datasets สำหรับ test
test_loader = data_utils.DataLoader(test_set, batch_size=batch_size, shuffle=True)  # สร้าง dataloader สำหรับ test set

# training_set = ConcatDataset([train_loader.dataset])
# testing_set = ConcatDataset([test_loader.dataset])


# Test loss and accuracy

train_score = []
test_score = []
training_loss = []

# F1-score
test_f1 = 0
test_precision = 0
test_recall = 0
f1_score_list = []
precision_list = []
recall_list = []
targets_fold = []
predicted_fold = []

# start training
print(lines)
print('Start training...')
# round = [i for i in range(1, 10)]
# round = [i for i in range(1)]
time_training = []

now=datetime.now()
dt_string=now.strftime("%d-%m-%Y_%H-%M-%S")

rounds = 1
for i in range(rounds):
    file_name = f'./model_save/CNN--Batch_size-{batch_size}--Learning_rate-{learning_rate}--Layer_size--{layer_size}--Epochs--{num_epochs}--round--{rounds}--{dt_string}'
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    print(f'File name: {file_name}')

    # Open the text file for writing
    with open(os.path.join(file_name, 'training_details.txt'), 'w') as file:
        now = datetime.now()
        dt_string = now.strftime("%d-%m_%H-%M-%S")
        print(f'Start {rounds} at {dt_string}')

        start_time_round = time.time()
        epochList = []
        lossList = []
        trainAccList = []
        testAccList = []

        # Define your CNN model and optimizer outside the epoch loop
        model = CNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            start_time = time.time()
            epochList.append(epoch)
            train_loss = 0.0
            train_total = 0
            train_correct = 0
            times = []

            # Training loop
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                model.train()
                outputs = model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()

            # Testing loop
            test_acc = 0.0
            test_total = 0.0
            model.eval()
            with torch.no_grad():
                for inputs, y_true in test_loader:
                    inputs, y_true = inputs.to(device), y_true.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    test_acc += (predicted == y_true).sum().item()
                    test_total += y_true.size(0)

                    # แปลงข้อมูลที่อยู่ใน GPU กลับเป็น NumPy arrays
                    y_true = y_true.cpu().numpy()
                    predicted = predicted.cpu().numpy()

                    # คำนวณค่า F1 score, precision, และ recall โดยใช้ scikit-learn functions. Zero division parameter ถูกใช้เพื่อป้องกันการหารด้วยศูนย์ถ้าไม่มีข้อมูลจริงในคลาสที่กำลังคำนวณ
                    f1 = f1_score(y_true, predicted, average='weighted')
                    precision = precision_score(y_true, predicted, average='weighted', zero_division=1)
                    recall = recall_score(y_true, predicted, average='weighted', zero_division=1)

                    # เพิ่มค่า F1 score, precision, และ recall ลงในลิสต์ที่เก็บผลลัพธ์
                    f1_score_list.append(f1)
                    precision_list.append(precision)
                    recall_list.append(recall)
                    targets_fold.append(y_true)
                    predicted_fold.append(predicted)

            endtime = time.time()
            usetime = endtime - start_time
            times.append(usetime)

            trainAccList.append(train_correct / train_total)
            testAccList.append(test_acc / test_total)
            if (train_loss / len(train_loader)) > 1.0:
                lossList.append(1)
            else:
                lossList.append(train_loss / len(train_loader))

            # Write to the text file
            if epoch % 1 == 0:
                TrainingLoss = train_loss / len(train_loader)
                Training_Accuracy = train_correct / train_total
                Testing_Accuracy = test_acc / test_total
                TimeMean = sum(times) / len(times)
                detail = (
                    f'Round {i + 1} \tEpoch: {epoch + 1}/{num_epochs} \tTraining Loss: {TrainingLoss:.10f} \tTraining Accuracy: {Training_Accuracy:.10f} \tTesting Accuracy: {Testing_Accuracy:.10f} \tTime Learning: {TimeMean:.3f} Sec. BatchSize {batch_size}\n')
                print(detail)
                file.write(detail)

            if epoch == num_epochs - 1:
                training_loss.append(TrainingLoss)
            test_f1 /= len(test_loader)
            test_precision /= len(test_loader)
            test_recall /= len(test_loader)

            # Calculate the average values
            average_f1 = np.average(f1_score_list)
            average_precision = np.average(precision_list)
            average_recall = np.average(recall_list)

            targets_tensor_list = [torch.from_numpy(target_array) for target_array in targets_fold]
            predicted_tensor_list = [torch.from_numpy(predicted_array) for predicted_array in predicted_fold]
            targets_tensor = torch.cat(targets_tensor_list, dim=0)
            predicted_tensor = torch.cat(predicted_tensor_list, dim=0)

        end_time_round = time.time()
        time_per_round = end_time_round - start_time_round
        time_left = time_per_round * (rounds - i - 1)
        print(f'Time round {i + 1}: {get_time_hh_mm_ss(time_per_round)} Sec. About time left {get_time_hh_mm_ss(round(time_left))}')

        print()
        print('F1-score: {:.4f}'.format(average_f1))
        print('Precision: {:.4f}'.format(average_precision))
        print('Recall: {:.4f}'.format(average_recall))
        print("report=\n", classification_report(targets_tensor, predicted_tensor))
        print()

        # Print Confusion Matrix
        classLabel = ['None Human', 'Less than or Equal to Three Humans', 'More than Three Humans']
        cm = confusion_matrix(targets_tensor, predicted_tensor)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="GnBu")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f"CNN Confusion Matrix(Test Normal) - Epochs: {num_epochs}, Learning Rate: {learning_rate}\n"
                f"Weight Decay: {weight_decay}, Batch Size: {batch_size}, Optimizer: {optimizer}, Enable BatchNorm2d")
        plt.savefig(os.path.join(file_name, 'CNN_confusion_matrix.png'), dpi=300)
        plt.show()

file.close()


time.sleep(5)

plt.grid(visible=True, which='major', axis='both', c='0.95', ls='-', linewidth=1.0, zorder=0)
plt.title(
    f"CNN=Learning_late={learning_rate},batch_size={batch_size},epoch={num_epochs},\nweight_decay={weight_decay},Layer_size={layer_size},cnnOptim={type(optimizer).__name__}")
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