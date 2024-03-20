import pandas as pd
import torch
from torch import nn
import firebase_admin
from firebase_admin import credentials, db
from tqdm import tqdm
import time
import numpy as np
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI,HTTPException, Request
from fastapi.staticfiles import StaticFiles



print('---------------------------------------------------------------------')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

input_size = 4 * 4 * 32  # Adjusted based on your CNN architecture
hidden_size = 100
layer_size = 1
num_classes = 2

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, stride=1, padding=0)
        self.batch1 = torch.nn.BatchNorm2d(16)
        self.drop2 = torch.nn.Dropout2d(0.2) 
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=0)
        self.batch2 = torch.nn.BatchNorm2d(32)
        self.drop2 = torch.nn.Dropout2d(0.2) 
        self.relu2 = torch.nn.ReLU()
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
        return out

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, hidden_size, layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_size, x.size(0), self.hidden_size).requires_grad_().to(device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_size, x.size(0), self.hidden_size).requires_grad_().to(device)

        # One time step
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])
        return out


print('starting process_data function')

# Initialize your models
cnn_model = CNN().to(device)
lstm_model = LSTM().to(device)

# Load the trained model weights
checkpoint = torch.load(r"C:\read_thermal\Clund_run_docker\model_and_optimizers.pth", map_location=device)
cnn_model.load_state_dict(checkpoint['cnn_model'])
lstm_model.load_state_dict(checkpoint['lstm_model'])

cnn_model.eval()
lstm_model.eval()

# Initialize Firebase Admin SDK
cred = credentials.Certificate(r"C:\read_thermal\Clund_run_docker\handdetect-47529-firebase-adminsdk-24820-86efffd565.json")
firebase_admin.initialize_app(cred, {"databaseURL": "https://handdetect-47529-default-rtdb.firebaseio.com/"})
# Initialize FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


async def process_data():
    try:
        timesteps = 180
        data_list = []

        # Get Data from Firebase
        firebase_data = db.reference("TEST").order_by_key().limit_to_last(timesteps).get()
        for index in range(timesteps):
            last_key = list(firebase_data.keys())[index]
            last_value = firebase_data[last_key]
            data_list.append(last_value)

        df = pd.DataFrame(data_list)
        
        temperatures = df.iloc[0,0:].str.split(',')
        total_temperatures = sum(len(temp) for temp in temperatures)

        print("Total temperatures in the DataFrame:", total_temperatures)

        video_frame = []
        data_num = 0
        X_img = []

        for line in tqdm(range(len(df))):
            if data_num < timesteps:
                frame = np.array(df.iloc[line, 0].split(','), dtype=int).reshape(8, 8)
                video_frame.append(frame)
                data_num += 1
            else:
                X_img.append(video_frame)
                video_frame = []
                data_num = 0

        if video_frame:
            X_img.append(video_frame)

        for i, input_frame in enumerate(X_img):
            input_tensor = torch.FloatTensor(input_frame).unsqueeze(0).unsqueeze(2).to(device)
            cnn_features = cnn_model(input_tensor)
            outputs = lstm_model(cnn_features)
            prediction_result = torch.argmax(outputs, dim=1).item()
            print(prediction_result)
            
            # Display prediction result
            if prediction_result == 1:
                result = "Normal"
            elif prediction_result == 0:
                result = "Alert ! Hand Requested !"
                
            return {"prediction_result": prediction_result, "result": result}
        

        
    except Exception as e:
        print("Error:", e)
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    try:
        # Call the process_data function to get prediction_result and result
        data = await process_data()
        prediction_result = data["prediction_result"]
        result = data["result"]
    except Exception as e:
        # Handle errors from process_data() function
        prediction_result = "Error"
        result = str(e)

    return templates.TemplateResponse(
        "index.html", {"request": request, "prediction_result": prediction_result, "result": result}
    )
    


    #         # Define an endpoint to trigger the process_data() function
    # @app.get("/")
    # async def trigger_process_data():
    #     return await process_data()