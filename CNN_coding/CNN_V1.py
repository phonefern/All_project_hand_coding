import numpy as np
import torch
import pandas as pd
from _plotly_utils import data_utils
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, random_split
import torch.utils.data as data_utils
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report, \
    ConfusionMatrixDisplay
import datetime


# -------------------- CNN Model --------------------
class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution 1 and Max pool 1
        self.cnn1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.batch1 = torch.nn.BatchNorm2d(32)
        self.drop1 = torch.nn.Dropout(0.1)
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=1)

        # Convolution 2 and Max pool 2
        self.cnn2 = torch.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batch2 = torch.nn.BatchNorm2d(128)
        self.drop2 = torch.nn.Dropout(0.2)
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=1)

        # Fully connected 1 (readout)
        self.flat = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(27 * 19 * 128, 256)
        self.fc2 = torch.nn.Linear(256, 64)
        self.fc3 = torch.nn.Linear(64, 3)

    def forward(self, x):
        # Convolution 1 and Max pool 1
        out = self.cnn1(x)
        out = self.batch1(out)
        out = self.drop1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)

        # Convolution 2 and Max pool 2
        out = self.cnn2(out)
        out = self.batch2(out)
        out = self.drop2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        # Resize
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.flat(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out


# -------------------- Select Device for Train Model --------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("-------------------- Select Device for Train Model --------------------")
print(f"Device use: {device}")
print()

# -------------------- Data Preprocessing --------------------

# list for the training set
X_train = []
X_test = []

y_train = []
y_test = []

data_path1 = "D:\\SP_DataUseTrainTest\\TrainTestData\\TrainData\\TrainNormalData.csv"
data_path2 = "D:\\SP_DataUseTrainTest\\TrainTestData\\TestData\\TestNormalData.csv"

data1 = pd.read_csv(data_path1)
data2 = pd.read_csv(data_path2)

X = data1.iloc[:, 1:769].values
y = data1["label"].values

X_train = data1.iloc[:, 1:769].values
y_train = data1["label"].values

X_test = data2.iloc[:, 1:769].values
y_test = data2["label"].values

# dataPath = "D:\\SP_TrainData\\TrainDataReady.csv"
# data = pd.read_csv(dataPath)
# X = data.iloc[:, 1:769].values
# y = data["label"].values
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_counterTrain = Counter(y_train)
y_counterTest = Counter(y_test)
print("-------------------- Ratio Train&Test Data --------------------")
print(f'Ratio Train Data is {y_counterTrain}')
print(f'Ratio Test Data is {y_counterTest}')
print()

# แปลงรูปของดาต้าให้อยู่ในรูปแบบของ tensor
x_trainImages = []
x_testImages = []
y_trainLabels = []
y_testLabels = []

# Plot Heatmap and save to jpg
width = 32
height = 24

Rawdata_train = [X_train, X_test]

data = [x_trainImages, x_testImages]

num = 0
for d in Rawdata_train:
    # Data Transforming
    for i in d:
        frame2D = []
        for h in range(24):
            frame2D.append([])
            for w in range(32):
                t = i[h * 32 + w]
                frame2D[h].append(t)

        data[num].append([frame2D])

    num += 1

x_trainImages = torch.FloatTensor(x_trainImages)
y_trainLabels = torch.LongTensor(y_train)
x_testImages = torch.FloatTensor(x_testImages)
y_testLabels = torch.LongTensor(y_test)

# Data Loader
print("-------------------- Train&Test Data Size --------------------")
print('Raw Training Images size: ', len(X_train))
print('Raw Training Label size: ', len(y_train))
# print('Raw Training Label size: ', len(y_train), ' | Unique Label: ', np.unique(np.array(X_train)))
print('Raw Testing Images size: ', len(X_test))
print('Raw Testing Label size: ', len(y_test), ' | Unique Label: ', np.unique(np.array(y_test)))
print()
print('Transformed X_trainImages Images size: ', x_trainImages.size())
print('Transformed Y_trainLabels Labels size: ', y_trainLabels.size())
print('Transformed X_testImages Images size: ', x_testImages.size())
print('Transformed Y_testLabels Labels size: ', y_testLabels.size())
print()

# Call CNN Model and Setting Hyperparameter
model = CNNModel().to(device)
criterion = torch.nn.CrossEntropyLoss()
num_epochs = 100
learning_rate = 0.0001
weight_decay = 0.0001
batch_size = 32

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

cnnOptimizer = ""
if isinstance(optimizer, torch.optim.Adam):
    cnnOptimizer = "Adam"
elif isinstance(optimizer, torch.optim.SGD):
    cnnOptimizer = "SGD"

# get the training set
train_set = data_utils.TensorDataset(x_trainImages, y_trainLabels)
train_loader = data_utils.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# get the test set
test_set = data_utils.TensorDataset(x_testImages, y_testLabels)
test_loader = data_utils.DataLoader(test_set, batch_size=batch_size, shuffle=True)

# training_set = ConcatDataset([train_loader.dataset])
# testing_set = ConcatDataset([test_loader.dataset])

# Test loss and accuracy
train_score = []
test_score = []
loss_list = []

# F1-score
test_f1 = 0
test_precision = 0
test_recall = 0
f1_score_list = []
precision_list = []
recall_list = []
targets_fold = []
predicted_fold = []

# -------------------- Setting for Graph --------------------
dt = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
dt_str = dt.replace(":", ".")
text = (f"CNN - Epochs {num_epochs}, LR {learning_rate} WeightDecay {weight_decay}, BatchSize {batch_size}, "
        f"Optimizer {cnnOptimizer}, Enable BatchNorm2d")

plt.rcParams['font.family'] = 'TH SarabunPSK'
plt.rcParams['font.size'] = 13

# -------------------- Train&Test CNN Model --------------------
print("-------------------- Train&Test CNN Model --------------------")
print('Start Training...')
print()
roundTrain = [i for i in range(1, 2)]

for i in roundTrain:

    # วนลูปที่กำหนดจำนวนครั้งที่ต้องการทำการฝึก (epochs) โมเดล
    for epoch in range(num_epochs):
        train_loss = 0.0  # กำหนดตัวแปรสำหรับเก็บค่า loss
        train_total = 0  # จำนวนข้อมูลทั้งหมดในการฝึก
        train_correct = 0  # จำนวนข้อมูลที่ถูกต้องในการฝึก

        # วนลูปที่ใช้กับ DataLoader ของชุดข้อมูลการฝึก, โดยแต่ละรอบจะโหลดข้อมูลแบบ batch จาก train_loader
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)  # ส่งข้อมูลและเป้าหมาย ไปยังอุปกรณ์ที่ใช้ GPU สำหรับการคำนวณ
            optimizer.zero_grad()  # ล้าง gradient ของตัวแปรโมเดล
            model.train()  # กำหนดโมเดลให้ทำงานในโหมด train
            outputs = model(data)  # รับค่า output จากโมเดล
            loss = criterion(outputs, target)  # คำนวณค่า loss โดยใช้ค่า output จากโมเดลและเป้าหมาย
            loss.backward()  # คำนวณ gradient ของค่า loss
            optimizer.step()  # ปรับปรุงค่า weights ของโมเดล
            train_loss += loss.item()  # บวกค่า loss ไปยังค่า train_loss ที่เก็บ
            _, predicted = torch.max(outputs.data, 1)  # คำนวณค่าทำนายของโมเดล
            train_total = target.size(0)  # นับจำนวนข้อมูลทั้งหมดใน batch
            train_correct = (predicted == target).sum().item()  # นับจำนวนข้อมูลที่ถูกต้องใน batch

        test_acc = 0.0  # กำหนดตัวแปรสำหรับเก็บค่าความถูกต้องของการทดสอบ
        test_total = 0.0  # กำหนดตัวแปรสำหรับเก็บจำนวนข้อมูลทั้งหมดในการทดสอบ
        model.eval()  # กำหนดโมเดลให้ทำงานในโหมด evaluation (ไม่ทำการปรับปรุง weights) เป็นการบอก model ว่าเราจะทำการ test แล้ว
        with torch.no_grad():  # ปิดการติดตาม gradient สำหรับการทดสอบเนื่องจากเราไม่ต้องการปรับปรุง weights ในการทดสอบ
            for batch_idx, (inputs, y_true) in enumerate(test_loader):  # วนลูปที่ใช้กับ DataLoader ของชุดข้อมูลการทดสอบ
                # Extracting images and target labels for the batch being iterated
                inputs, y_true = inputs.to(device), y_true.to(device)  # ส่งข้อมูลและเป้าหมายไปยังอุปกรณ์ที่ใช้ (GPU หรือ CPU) สำหรับคำนวณ

                # Calculating the model output and the cross entropy loss
                outputs = model(inputs)  # รับค่า output จากโมเดล

                # Calculating the accuracy of the model
                _, predicted = torch.max(outputs.data, 1)  # คำนวณค่าทำนายของโมเด
                test_acc += (predicted == y_true).sum().item()  # นับจำนวนข้อมูลที่ถูกต้องในการทดสอบและบวกไปยัง test_acc
                test_total += y_true.size(0)  # นับจำนวนข้อมูลทั้งหมดใน batch และบวกไปยัง test_total

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

        # คำนวณค่า accuracy และ loss ของ train set และ test set
        if epoch % 2 == 0:  # ตรวจสอบว่า epoch เป็นค่าคู่หรือไม่
            # คำนวณค่า loss และ accuracy สำหรับการฝึกและการทดสอบ
            TrainingLoss = train_loss / len(train_loader)
            Training_Accuracy = train_correct / train_total
            Testing_Accuracy = test_acc / test_total
            print(
                'Round {} \tEpoch: {}\{} \tTraining Loss: {:.10f} \tTraining Accuracy: {:.10f} \tTesting Accuracy: {:.10f}'.format(
                    i, epoch + 2, num_epochs, TrainingLoss, Training_Accuracy, Testing_Accuracy))

            # เพิ่มค่า loss และ accuracy ของการฝึกและการทดสอบลงในลิสต์ที่เก็บผลลัพธ์
            loss_list.append(TrainingLoss)
            train_score.append(Training_Accuracy)  # สำหรับเก็บค่า accuracy ของ train
            test_score.append(Testing_Accuracy)  # สำหรับเก็บค่า accuracy ของ test

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

    # Print the F1-score, precision, and recall
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
              f"Weight Decay: {weight_decay}, Batch Size: {batch_size}, Optimizer: {cnnOptimizer}, Enable BatchNorm2d")
    plt.savefig(f"D:\\ML\\GraphCNN\\Confusion Matrix {dt_str}-{text}.png", dpi=300)
    plt.show()

    plt.plot(loss_list, label='Loss', alpha=0.8, linewidth=1.0, color="#75E6DA")
    plt.plot(test_score, label='Test Accuracy', alpha=0.8, linewidth=1.0, color="#189AB4")
    plt.plot(train_score, label='Train Accuracy', alpha=0.8, linewidth=1.0, color="#05445E")
    plt.grid(visible=True, which='major', axis='both', c='0.95', ls='-', linewidth=0.5, zorder=0)
    plt.legend(loc='upper right', fontsize=10)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.title(f"CNN(Test Normal) - Epochs: {num_epochs}, Learning Rate: {learning_rate}\n"
              f"Weight Decay: {weight_decay}, Batch Size: {batch_size}, Optimizer: {cnnOptimizer}, Enable BatchNorm2d")
    plt.savefig(f"D:\\ML\\GraphCNN\\Loss Accuracy {dt_str}-{text}.png", dpi=300)
    plt.show()

print("Finish.")
