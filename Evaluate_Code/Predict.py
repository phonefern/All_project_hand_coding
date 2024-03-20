import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pylab as plt
import os
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
        self.drop1 = torch.nn.Dropout2d(0.2)  # **optional
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
        self.drop2 = torch.nn.Dropout2d(0.2)  # **optional
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
layer_size = 1  # Hidden Layers

# num_classes 
num_classes = 2

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
        print(x.shape)

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

print(device)

# Load your trained models and checkpoint from the specific path
load_path = 'C:\\read_thermal\\Model\\model_and_optimizers.pth'
checkpoint = torch.load(load_path)

# Initialize your models
cnn_model = CNN()
lstm_model = LSTM()

# Load the trained model weights
cnn_model.load_state_dict(checkpoint['cnn_model'])
lstm_model.load_state_dict(checkpoint['lstm_model'])

# Set the models to evaluation mode
cnn_model.eval()
lstm_model.eval()

# Preprocess and prepare the data
timesteps = 180
width = 8
height = 8

def preprocess_data(df):
    X, Y = [], []

    video_frame = []
    data_num = 0

    for line in tqdm(range(len(df))):
        if data_num < timesteps:
            frame2D = []
            frame = df.iloc[line, 0:-1]
            for h in range(height):
                frame2D.append([])
                for w in range(width):
                    t = frame[h * width + w]
                    frame2D[h].append(t)

            video_frame.append([frame2D])
            data_num = data_num + 1
        else:
            Y.append(df.iloc[line, -1])
            X.append(video_frame)
            video_frame = []
            data_num = 0
    # print(video_frame)
    X, Y = torch.FloatTensor(X).to(device), torch.LongTensor(Y).to(device)
    print(X.shape)
    return X, Y


new_data = pd.read_csv(r'C:\read_thermal\sumdata_2024\data_summary\evaluate_20000_predict_Noise_0.csv').dropna().drop("TimeStamp", axis=1)

# Preprocess the new data
X_new, Y_new = preprocess_data(new_data)

# Initialize your models
cnn_model = CNN()
lstm_model = LSTM()

# Load the trained model weights
checkpoint = torch.load(f'C:\\read_thermal\Model\CNN_LSTM\model_and_optimizers.pth')  # Replace with the actual model path
cnn_model.load_state_dict(checkpoint['cnn_model'])
lstm_model.load_state_dict(checkpoint['lstm_model'])


test_acc = 0.0
test_total = 0.0
# Set the models to evaluation mode
cnn_model.eval()
lstm_model.eval()


epochList = []
lossList = []
trainAccList = []
testAccList = []

# F1-score
test_f1 = 0
test_precision = 0
test_recall = 0
f1_score_list = []
precision_list = []
recall_list = []
targets_fold = []
predicted_fold = []

# Define a mapping for labels with integer keys

# Make predictions
with torch.no_grad(): # Turn off gradients for validation, saves memory and computations
    cnn_model.to(device)
    lstm_model.to(device)
    X_new = X_new.to(device)  # Move the input data to the device
    # Pass data through the CNN model
    cnn_features = cnn_model(X_new)
    # Pass CNN features through the LSTM model
    outputs = lstm_model(cnn_features)

    _, predicted_class = torch.max(outputs, 1)
    test_acc += (predicted_class == Y_new).sum().item()  # นับ accuracy
    test_total += Y_new.size(0)

    # แปลงข้อมูลที่อยู่ใน GPU กลับเป็น NumPy arrays
    Y_new = Y_new.cpu().numpy()
    predicted_class = predicted_class.cpu().numpy()

    # คำนวณค่า F1 score, precision, และ recall โดยใช้ scikit-learn functions. Zero division parameter ถูกใช้เพื่อป้องกันการหารด้วยศูนย์ถ้าไม่มีข้อมูลจริงในคลาสที่กำลังคำนวณ
    f1 = f1_score(Y_new, predicted_class, average='weighted')
    precision = precision_score(Y_new, predicted_class, average='weighted', zero_division=1)
    recall = recall_score(Y_new, predicted_class, average='weighted', zero_division=1)

    # เพิ่มค่า F1 score, precision, และ recall ลงในลิสต์ที่เก็บผลลัพธ์
    f1_score_list.append(f1)
    precision_list.append(precision)
    recall_list.append(recall)
    targets_fold.append(Y_new)
    predicted_fold.append(predicted_class)





# Define a mapping for labels
label_mapping = {
    0: "0-normal",
    1: "1-abnormal"
}




# Calculate the length of the predicted_class tensor without CUDA
total_predict = len(torch.from_numpy(predicted_class).cpu())


accuracies = {label: {'correct': 0, 'total': 0} for label in label_mapping.values()}

for i in range(total_predict): # วนลูปเพื่อนับความถูกต้องของการทำนาย และเก็บข้อมูลเพื่อคำนวณความแม่นยำ
    actual_label = Y_new[i] # ค่าจริง ที่อยู่ใน Y_new
    predicted_label = label_mapping[predicted_class[i].item()]

    is_correct = actual_label == predicted_class[i].item()
    # accuracies[predicted_label]['total'] += 1
    if is_correct:
        accuracies[predicted_label]['correct'] += 1

    print(f"Video Frame ที่ {i+1} ทำนายเป็น '{predicted_label}' ของจริง '{actual_label}' ความเเม่นยำ --> {'ถูก' if is_correct else 'ผิด'}")

correct_predictions = 0
total_predictions = len(predicted_class)

for i in range(total_predictions):
    actual_label = Y_new[i].item()
    predicted_label = predicted_class[i].item()

    if actual_label == predicted_label:
        correct_predictions += 1

overall_accuracy = (correct_predictions / total_predictions) * 100
print(f"Overall Accuracy: {overall_accuracy:.2f}%")



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

result_summary = (
    f'Average F1-score: {average_f1:.4f}\n'
    f'Average Precision: {average_precision:.4f}\n'
    f'Average Recall: {average_recall:.4f}\n'
    f'Classification Report:\n{classification_report(targets_tensor, predicted_tensor)}\n'
        )

# file_name = r'C:\read_thermal\Model\Evaluation_Model_predict\plots'
file_name = ''
# Print Confusion Matrix
classLabel = ['Normal ', 'Abnormal']
cm = confusion_matrix(targets_tensor, predicted_tensor, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classLabel)
disp.plot(cmap="GnBu", values_format='d')  # 'd' stands for integer format
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f"CNN-LSTM Confusion Matrix")
plt.savefig(os.path.join(file_name, 'CNN-LSTM_confusion_matrix_plot.png'), dpi=300)
plt.show()
print("Save plot success")

# print(f'\n acc: {acc} sum {sum} -> {acc/sum*100}%')



# print('Starting Predict')
# print('-----------------------------------------------------------------------------------------------')
# print(f'Predicted labels: {predicted_labels}')