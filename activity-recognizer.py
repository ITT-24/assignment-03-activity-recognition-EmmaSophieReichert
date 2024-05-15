# this program recognizes activities

import matplotlib
from matplotlib import pyplot as plt
from sklearn import svm
import seaborn as sns # for nice visualizations
import pandas as pd # for loading the data from csv
import numpy as np
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
from sklearn import svm
import os
from typing import Dict, List
from scipy import signal

FOLDER_PATH = "data"

ACTIVITIES = ["running", "rowing", "lifting", "jumpingjacks"]
COLUMNS = ["activity", "acc_x_fr", "acc_y_fr", "acc_z_fr", "gyro_x_fr", "gyro_y_fr", "gyro_z_fr"]

KERNEL_SIZE = 20
KERNEL_SIGMA = 5
SAMPLING_RATE = 100

#--- methods from assignment 2

def apply_kernel(data):
    kernel = signal.windows.gaussian(KERNEL_SIZE, KERNEL_SIGMA) # create a kernel
    kernel /= np.sum(kernel) # normalize the kernel so it does not affect the signal's amplitude
    return np.convolve(data, kernel, 'same') # apply the kernel to the signal

def apply_hamming_window(data):
    hamming_window = np.hamming(len(data))
    return data * hamming_window

def get_max_frequency(data) -> float:
    
    #drop all NaN, because spectrum is NaN otherwise
    mask = ~np.isnan(data)
    data = data[mask]
    
    # calculate spectrum using a fast fourier transform
    spectrum = np.abs(np.fft.fft(data))
    
    # resample x axis of spectrum to match frequency even if sample_length != 1
    frequencies = np.fft.fftfreq(len(data), 1/SAMPLING_RATE)
    
    # get rid of negative half
    mask = frequencies >= 0
    positive_frequencies = frequencies[mask]
    spectrum = spectrum[mask]

    #plt.plot(spectrum)
    #plt.show(block=True)

    #get max frequency
    max_frequency = positive_frequencies[np.argmax(spectrum)]

   # print(np.argmax(spectrum))
    #max_frequency = frequencies[np.argmax(spectrum)]
    return max_frequency

def get_frequency(data) -> int:
    # Convert audio data to numpy array
    data = np.array(data)

    data = apply_kernel(data)

    #plt.plot(data)
    #plt.show(block=True)

    data = apply_hamming_window(data)
        
    # plt.plot(data)
    # plt.show(block=True)

    max_frequency = get_max_frequency(data)
    return max_frequency

#--- other helper methods

#from maschine_learning_tour.ipynb
def normalize_data(df) -> pd.DataFrame:
    scaled_samples = scale(df[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]])
    df_mean = df.copy()
    df_mean[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]] = scaled_samples
    return df_mean

class ActivityRecognizer:

    def __init__(self) -> None:
        self.preprocess_data()
        self.train_classifier()
        self.test_data()

    def preprocess_data(self) -> None:
        data = self.read_data()
        #self.normalize_data(data)
        data = self.apply_fft(data)
        print(data)

    #used GPT for this method
    def read_data(self) -> Dict[str, List[pd.DataFrame]]:
        running_dfs = []
        rowing_dfs = []
        lifting_dfs = []
        jumpingjacks_dfs = []
        
        # iterate over all files in folder
        for filename in os.listdir(FOLDER_PATH):
            file_path = os.path.join(FOLDER_PATH, filename)
            if 'running' in filename:
                running_dfs.append(normalize_data(pd.read_csv(file_path)))
            elif 'rowing' in filename:
                rowing_dfs.append(normalize_data(pd.read_csv(file_path)))
            elif 'lifting' in filename:
                lifting_dfs.append(normalize_data(pd.read_csv(file_path)))
            elif 'jumpingjacks' in filename:
                jumpingjacks_dfs.append(normalize_data(pd.read_csv(file_path)))
        
        return {
            "running": running_dfs,
            "rowing": rowing_dfs,
            "lifting": lifting_dfs,
            "jumpingjacks": jumpingjacks_dfs
        }

    #calculate the frequency for every measure of every dataframe and safe it in one dataframe
    def apply_fft(self, data) -> None:
        df_all = pd.DataFrame(columns=COLUMNS)
        for activity in ACTIVITIES:
            for df in data[activity]:
                acc_x_fr = get_frequency(df["acc_x"])
                acc_y_fr = get_frequency(df["acc_y"])
                acc_z_fr = get_frequency(df["acc_z"])
                gyro_x_fr = get_frequency(df["gyro_x"])
                gyro_y_fr = get_frequency(df["gyro_y"])
                gyro_z_fr = get_frequency(df["gyro_z"])
                df_all.loc[len(df_all)] = [activity, acc_x_fr, acc_y_fr, acc_z_fr, gyro_x_fr, gyro_y_fr, gyro_z_fr]
        return df_all

    def train_classifier(self) -> None:
        pass

    def test_data(self) -> None:
        pass

    def get_activity(self, data) -> str:
        return ""
    
activity = ActivityRecognizer()