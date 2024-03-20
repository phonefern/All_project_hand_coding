import os
from datetime import datetime, timedelta
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
import matplotlib.pylab as plt
import gc
from torch import nn
from pygame import mixer
from statistics import mean
import time
import inspect
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score

def main(batchsize, int_epoch, mod_epoch, int_round):
    print(f'Start: {__name__}, BatchSize {batchsize} ')

    def get_time_hh_mm_ss(sec):
        td_str = str(timedelta(seconds=sec))
        # split string into individual component
        x = td_str.split(':')
        x_str = f'hh:mm:ss: {x[0]} Hours {x[1]} Minutes {x[2]} Seconds'
        return x_str

    def play(path):
        mixer.init()
        mixer.music.load(path)
        mixer.music.play()
        if (path == 'D:\Code\HeatMapSerialRead\heart-stop.mp3'):
            time.sleep(0.3)
            mixer.music.stop()

    def Average(lst):
        return sum(lst) / len(lst)

    signal = 'D:\Code\HeatMapSerialRead\heart-stop.mp3'

    gc.collect()
    torch.cuda.empty_cache()

    layer_cnn = 2  # !!!
    dropout = "0.2"  # !!!

    torch.backends.cudnn.enabled = False

    # training_files = ['train2case']
    # testing_files = ['test2case']

    # training_files = ['train2caseMod3']
    # testing_files = ['test2caseMod3']

    # training_files = ['train2caseMod5']
    # testing_files = ['test2caseMod5']

    # training_files = ['train2caseMod2']
    # testing_files = ['test2caseMod2']

    # training_files = ['BtrainMakeItBalance']
    # testing_files = ['BtestMakeItBalance']

   
    # training_files = ['Training_2024_concat3.csv']
    # testing_files = ['Testing_2024_v2_concat3.csv']

    # training_files = ['AtrainOriginal']
    # testing_files = ['AtestOriginal']

    training_files = ['Training_2024_use.csv']
    testing_files = ['Testing_2024_v2.csv']

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    ### Preprocessing
    print('\n ---- Preprocessing Starting ---- ', '\n')

    timesteps = 180
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
            df = df = pd.read_csv(f'C:\\read_thermal\sumdata_2024\data_summary\\{i}')
            df.drop('TimeStamp', axis=1, inplace=True)
            df = df.dropna()
            X.append(df)
        return pd.concat(X, axis=0, ignore_index=True)

    def transformSequenceData(df):

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

    batch_size = batchsize  # [2,4,8,16,32,64,128,256] *Out of memory [512, 1024]
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
            # Input size: 32*24*1
            # Spatial extend of each one (kernelConv size), F = 3
            # Slide size (strideConv), S = 1
            # Padding, P = 2
            ## Width: ((32 - 3 + 2 * 2) / 1) + 1 = 34.0 #*# W2 = (( W1 - F + 2(P) ) / S ) + 1
            ## High: ((24 - 3 + 2 * 2) / 1) + 1 = 26.0 #*# H2 = (( H1 - F + 2(P) ) / S ) + 1
            ## Depth: 16
            ## Output Conv Layer1: 34.0 * 26.0 * 16
            self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, stride=1, padding=0)
            self.batch1 = torch.nn.BatchNorm2d(16)  # **optional
            self.drop1 = torch.nn.Dropout2d(0.2)  # **optional
            self.relu1 = torch.nn.ReLU()

            # Max Pooling Layer 1
            # Input size: 34.0 * 26.0 * 16
            ## Spatial extend of each one (kernelMaxPool size), F = 2
            ## Slide size (strideMaxPool), S = 2
            # Output Max Pooling Layer 1
            ## Width: ((34.0 - 2) / 2) + 1 = 17.0 #*# (( W2 - F ) / S ) + 1
            ## High: ((26.0 - 2) / 2) + 1 = 13.0 #*# (( H2 - F ) / S ) + 1
            ## Depth: 16
            ### Output Max Pooling Layer 1: 17.0 * 13.0 * 16
            self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=1)

            # Conv Layer 2
            # Input size: 17.0*13.0*16
            # Spatial extend of each one (kernelConv size), F = 3
            # Slide size (strideConv), S = 1
            # Padding, P = 2
            ## Width: ((17.0 - 3 + 2 * 2) / 1) + 1 = 19.0
            ## High: ((13.0 - 3 + 2 * 2) / 1) + 1 = 15.0
            ## Depth: 32
            ## Output Conv Layer2: 19.0 * 15.0 * 32
            self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=0)
            self.batch2 = torch.nn.BatchNorm2d(32)  # **optional
            self.drop2 = torch.nn.Dropout2d(0.2)  # **optional
            self.relu2 = torch.nn.ReLU()

            # Max Pooling Layer 2
            # Input size: 19.0 * 15.0 * 32
            ## Spatial extend of each one (kernelMaxPool size), F = 2
            ## Slide size (strideMaxPool), S = 2
            # Output Max Pooling Layer 2
            ## Width: ((19.0 - 2) / 2) + 1 = 9.5
            ## High: ((15.0 - 2) / 2) + 1 = 7.5
            ## Depth: 32
            ### Output Max Pooling Layer 2: 9.5 * 7.5 * 32
            self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=1)

            # Conv Layer 3
            # Input size: 9.5*7.5*32
            # Spatial extend of each one (kernelConv size), F = 3
            # Slide size (strideConv), S = 1
            # Padding, P = 2
            ## Width: ((9.5 - 3 + 2 * 2) / 1) + 1 = 11.5
            ## High: ((7.5 - 3 + 2 * 2) / 1) + 1 = 9.5
            ## Depth: 64
            ## Output Conv Layer3: 11.5 * 9.5 * 64
            # self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2)
            # self.batch3 = torch.nn.BatchNorm2d(64)  # **optional
            # # self.drop3 = torch.nn.Dropout2d(0.5)  # **optional
            # self.relu3 = torch.nn.ReLU()

            # Max Pooling Layer 3
            # Input size: 11.5 * 9.5 * 64
            ## Spatial extend of each one (kernelMaxPool size), F = 2
            ## Slide size (strideMaxPool), S = 2
            # Output Max Pooling Layer 3
            ## Width: ((11.5 - 2) / 2) + 1 = 5.75
            ## High: ((9.5 - 2) / 2) + 1 = 4.75
            ## Depth: 64
            ### Output Max Pooling Layer 3: 5.75 * 4.75 * 64
            # self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

            # # Conv Layer 4
            # # Input size: 5.75*4.75*64
            # # Spatial extend of each one (kernelConv size), F = 3
            # # Slide size (strideConv), S = 1
            # # Padding, P = 2
            # ## Width: ((5.75 - 3 + 2 * 2) / 1) + 1 = 7.75
            # ## High: ((4.75 - 3 + 2 * 2) / 1) + 1 = 6.75
            # ## Depth: 128
            # ## Output Conv Layer4: 7.75 * 6.75 * 128
            # self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2)
            # self.batch4 = torch.nn.BatchNorm2d(128)  # **optional
            # # self.drop4 = torch.nn.Dropout2d(0.5)  # **optional
            # self.relu4 = torch.nn.ReLU()

            # Max Pooling Layer 4
            # Input size: 7.75 * 6.75 * 128
            ## Spatial extend of each one (kernelMaxPool size), F = 2
            ## Slide size (strideMaxPool), S = 2
            # Output Max Pooling Layer 4
            ## Width: ((7.75 - 2) / 2) + 1 = 3.875
            ## High: ((6.75 - 2) / 2) + 1 = 3.375
            # ## Depth: 128
            # ### Output Max Pooling Layer 4: 3.875 * 3.375 * 128
            # self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

            # Conv Layer 5
            # Input size: 3.875*3.375*128
            # Spatial extend of each one (kernelConv size), F = 3
            # Slide size (strideConv), S = 1
            # Padding, P = 2
            ## Width: ((3.875 - 3 + 2 * 2) / 1) + 1 = 5.875
            ## High: ((3.375 - 3 + 2 * 2) / 1) + 1 = 5.375
            ## Depth: 256
            ## Output Conv Layer5: 5.875 * 5.375 * 256
            # self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=2)
            # self.batch5 = torch.nn.BatchNorm2d(256)  # **optional
            # # self.drop5 = torch.nn.Dropout2d(0.5)  # **optional
            # self.relu5 = torch.nn.ReLU()

            # Max Pooling Layer 5
            # Input size: 5.875 * 5.375 * 256
            ## Spatial extend of each one (kernelMaxPool size), F = 2
            ## Slide size (strideMaxPool), S = 2
            # Output Max Pooling Layer 5
            ## Width: ((5.875 - 2) / 2) + 1 = 2.9375
            ## High: ((5.375 - 2) / 2) + 1 = 2.6875
            ## Depth: 256
            ### Output Max Pooling Layer 5: 2.9375 * 2.6875 * 256
            # self.pool5 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

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

            # # # Conv 3
            # out = self.conv3(out)
            # out = self.batch3(out)
            # out = self.relu3(out)

            # # # Max Pool 3
            # out = self.pool3(out)

            # # # # # Conv 4
            # out = self.conv4(out)
            # out = self.batch4(out)
            # out = self.relu4(out)
            #
            # # # # # Max Pool 4
            # out = self.pool4(out)
            #
            # # # # Conv 5
            # out = self.conv5(out)
            # out = self.batch5(out)
            # out = self.relu5(out)
            #
            # # # # Max Pool 5
            # out = self.pool5(out)

            out = out.view(i.shape[0], i.shape[1], -1)
            # print(out.shape)
            return out

    # Customize the model LSTM

    # input_size = 9*7*32 # for 2 layers CNN
    # input_size = 5*4*64 # for 3 layers CNN
    input_size = 4 * 4 * 32 # for 4 layers CNN
    # input_size = 2*2*256 # for 5 layers CNN

    hidden_size = 100  # Hidden Unit
    layer_size = 1  # Hidden Layers

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
    num_epochs = int_epoch  # [50, 100, 150, 250, 300]
    # num_epochs = []
    learning_rate = 0.0001
    weight_decay = 0.0001

    criterion = torch.nn.CrossEntropyLoss()

    print('Start training...')

    time_training = []

    rounds = int_round

    for i in range(rounds):

        now = datetime.now()
        dt_string = now.strftime("%d-%m_%H-%M-%S")
        print(f'Start {rounds} at {dt_string}')

        start_time_round = time.time()
        epochList = []
        lossList = []
        trainAccList = []
        testAccList = []

        true_value = []
        pred_value = []

        print(lines)

        loss_num = 0


        cnn_model = CNN().to(device)
        for layer in cnn_model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # cnn_optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate, momentum=0.9)
        lstm_model = LSTM().to(device)
        lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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

                    true_value.append(y_true.cpu().numpy()[0])
                    pred_value.append(predicted_test.cpu().numpy()[0])

            print(f'Epoch: {epoch}\ {num_epochs}')
            print(f'Confusion Matrix:\n {confusion_matrix(true_value, pred_value)}')
            print(f'Accuracy Score: {accuracy_score(true_value, pred_value)}')
            print(f'Precision Score: {precision_score(true_value, pred_value, average="weighted")}')
            print(f'Recall Score: {recall_score(true_value, pred_value, average="weighted")}')
            print(f'F1 Score: {f1_score(true_value, pred_value, average="weighted")}')
            print(f'Classification Report:\n {classification_report(true_value, pred_value, digits=4)}')

            endtime = time.time()
            usetime = endtime - start_time
            times.append(usetime)
            trainAccList.append(train_correct / train_total)
            testAccList.append(test_acc / test_total)

            if (train_loss / len(train_loader)) > 1.0:
                lossList.append(1)
            else:
                lossList.append(train_loss / len(train_loader))

            # คำนวณค่า accuracy และ loss ของ train set และ test set
            if epoch % mod_epoch == 0:
                TrainingLoss = train_loss / loss_num
                Training_Accuracy = train_correct / train_total
                Testing_Accuracy = test_acc / test_total
                TimeMean = mean(times)

                print(
                    'Round {} \tEpoch: {}\{} \tTraining Loss: {:.10f} \tTraining Accuracy: {:.10f} \tTesting Accuracy: {:.10f} \tTime Learning {:.3f} Sec. BatchSize {}'.format(
                        i + 1, epoch + 1, num_epochs, TrainingLoss, Training_Accuracy, Testing_Accuracy, TimeMean,
                        batch_size))
                time_training.append(TimeMean)

            if epoch == num_epochs - 1:
                training_loss.append(TrainingLoss)

        print(f'\n------------------ Evaluating Model ------------------')

        # Save the trained model and optimizer
        torch.save({'cnn_model': cnn_model.state_dict(),
                    'lstm_model': lstm_model.state_dict(),
                    'cnn_optimizer': cnn_optimizer.state_dict(),
                    'lstm_optimizer': lstm_optimizer.state_dict()}, 'model_and_optimizers.pth')

        end_time_round = time.time()
        time_per_round = end_time_round - start_time_round
        time_left = time_per_round * (rounds - i - 1)
        print(
            f'Time round {i + 1}: {get_time_hh_mm_ss(time_per_round)} Sec. About time left {get_time_hh_mm_ss(round(time_left))}.')
        train_score.append(Training_Accuracy)
        test_score.append(Testing_Accuracy)

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

    folder = f'./plots/{dt_string}-{layer_cnn}layerCNN-{dropout}-{layer_size}hiddenLayerlstm-{hidden_size}hiddenUnitlstm-{type(cnn_optimizer).__name__}cnnOpti-{type(lstm_optimizer).__name__}lstmOpti-{batch_size}batchSize-{rounds}round'
    os.makedirs(folder, exist_ok=True)

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
    print("Save plot success")

    codeCNN = inspect.getsource(CNN)

    summary = f"""Files = {training_files + testing_files}, ({lengthXtrain}, {lengthXtest}),\n
    TimeSteps = {timesteps},\n
    SizeFiles / TimeSteps = {lengthXtrain / timesteps}, {lengthXtest / timesteps},\n
    batch_size={batch_size},\n
    epoch={num_epochs},\n
    cnnOptim={type(cnn_optimizer).__name__},\n
    lstmOptim={type(lstm_optimizer).__name__},\n
    CNN Customization :\n{codeCNN},\n
    LSTM Customization :\nInputSize={str(input_size)}\nHiddenLayers={layer_size}\nHiddenUnits={hidden_size},\n
    learning_rate={learning_rate},\n
    weight_decay={weight_decay},\n
    timesteps={timesteps},\n
    hidden_size={hidden_size},\n
    layer_size={layer_size},\n
    meanLoss={mean(training_loss):.2f},\n
    TrainAcc: Mean{mean(train_score):.2f}, Max{max(train_score):.2f}, Min{min(train_score):.2f},\n
    TestAcc: Mean{mean(test_score):.2f}, Max{max(test_score):.2f}, Min{min(test_score):.2f},\n
    TimeTraining = Mean{mean(time_training):.2f}, Max{max(time_training):.2f}, Min{min(time_training):.2f} SEC."""

    plt.savefig(
        f'./plots/{dt_string}-{layer_cnn}layerCNN-{hidden_size}hiddenUnitlstm-{layer_size}hiddenLayerlstm-{mean(training_loss):.2f}MeanLoss-{mean(train_score):.2f}MeanTrainAcc-{mean(test_score):.2f}MeanTestAcc.png')

    with open(f'{folder}/{dt_string}.txt', 'x') as w:
        w.write(summary)
        w.close()

    plt.clf()
    plt.close()

    print("--------------- finish ----------------")
    play(signal)

main(batchsize=64,int_epoch=100,mod_epoch=1,int_round = 1)