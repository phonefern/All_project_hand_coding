import pandas as pd
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_out):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers

        self.batch_norm = nn.BatchNorm1d(input_size)  # BatchNorm1d for input features

        layers = []
        layers.append(self.batch_norm)
        layers.append(nn.Linear(self.input_size, self.hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=drop_out))

        for i in range(len(self.hidden_layers) - 1):
            layers.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(self.hidden_layers[i + 1]))  # BatchNorm1d for hidden layers
            layers.append(nn.Dropout(p=drop_out))

        layers.append(nn.Linear(self.hidden_layers[-1], self.output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class_names = ['Very Uncomfortable','Uncomfortable','Neutral','Comfortable' ,'Very Comfortable']

file_path = r"C:\Users\GUITAR\PycharmProjects\Project\Preprocessing\split_hrv_test_complete.csv"
data = pd.read_csv(file_path)

target_features = ['LFHF', 'Normalized_EDA', 'BodyTemp_IR', 'bmi']
selected_data = data[target_features]
drop_out = 0.0
input_size = len(target_features)
hidden_layers = [400]  # จำนวน hidden units
output_size = 5  # จำนวน output units

model = NeuralNetwork(input_size, output_size, hidden_layers, drop_out)

model.load_state_dict(torch.load('final-model.pth', map_location=torch.device('cpu')))
model.eval()

with torch.no_grad():
    inputs = torch.tensor(selected_data.values, dtype=torch.float)
    predictions = model(inputs)
    # print(predictions.tolist())
check = 0
acc = 0
sum = 0
class_index = torch.argmax(predictions, dim=1)  # หาคลาสที่มีค่าสูงสุด
for i in range(len(class_index)):
    sum = sum + 1
    if class_names[class_index[i]] == data['ComfortLevel'][i]:
        check = 1
        acc = acc + 1
    else:
        check = 0
    print(f"ข้อมูลตัวที่ {i+1} ทำนายเป็น '{class_names[class_index[i]]}' ของจริง '{data['ComfortLevel'][i]}' ได้คะแนนไป --> {check}")
print(f'\n acc: {acc} sum {sum} -> {acc/sum*100}%')