import serial
import time
import datetime
import pytz
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class SerialDataReader:
    def __init__(self, port, baud_rate=115200, timeout=0.1):
        self.ser = serial.Serial(port, baud_rate, timeout=timeout)
        time.sleep(0.2)
    def read_line(self):
        return self.ser.readline().decode('utf-8')

class HeatmapPlotter:
    def __init__(self):
        plt.ion()  # Enable interactive mode
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.im = None
        self.current_data = None  # Store the current data

    def update_heatmap(self, data, timestamp):
        data_list = [float(x) for x in data]  # Convert data to floats
        frame2D = np.array(data_list).reshape(8, 8)

        if self.im is None:
            # If self.im is None, create the initial plot
            self.im = self.ax.imshow(frame2D, cmap="coolwarm", vmin=22, vmax=29)
            self.ax.set_title(f"Heatmap at Time: {timestamp}")

            # Add thermal labels
            for i in range(8):
                for j in range(8):
                    self.ax.text(j, i, str(frame2D[i, j]), ha='center', va='center', color='black', fontsize=5)

            # Add a colorbar
            divider = make_axes_locatable(self.ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(self.im, cax=cax)
        else:
            # If self.im exists, update the existing plot
            self.im.set_data(frame2D)
            self.ax.set_title(f"Heatmap at Time: {timestamp}")

        # Store the current data
        self.current_data = data_list

        # Refresh the plot and introduce a short delay to display it
        plt.pause(0.1)

    def close(self):
        plt.ioff()  # Disable interactive mode
        plt.show()  # Keep the plot window open

    def get_current_data(self):
        return self.current_data

def get_timestamp():
    # Replace this with your code to get the timestamp for the data
    # Return a timestamp in the format 'YYYY-MM-DD HH:MM:SS.sss'
    pass

def main():
    # Initialize data reader and heatmap plotter
    data_reader = SerialDataReader('COM8')  # Replace 'COM10' with your serial port
    heatmap_plotter = HeatmapPlotter()

    # Define the Thailand time zone
    thailand_tz = pytz.timezone('Asia/Bangkok')

    try:
        while True:
            line = data_reader.read_line()
            timestamp_thailand = datetime.datetime.now(thailand_tz).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            print(f"TimeStamp: [{timestamp_thailand}]")

            start_index = line.find("[")
            end_index = line.find("]")

            if start_index != -1 and end_index != -1:
                data_part = line[start_index+1:end_index]
                data_list = data_part.split(',')

                if len(data_list) < 64:
                    missing_elements = 64 - len(data_list)
                    data_list.extend(['0'] * missing_elements)

                # Make sure data_list is not None
                if data_list is not None:
                    heatmap_plotter.update_heatmap(data_list, timestamp_thailand)

            time.sleep(0.01)

    except KeyboardInterrupt:
        heatmap_plotter.close()

if __name__ == "__main__":
    main()
