import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

#* TRANSFORMING DATASETS TO FILTER DATA *#
# Function to filter datasets:
def filter_datasets(input_directory: str, output_directory: str, lowcut: float, highcut: float, frecuency: float):

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Filter coeficents:
    nyquist = frecuency / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype="band")

    # Applies the filter to each dataset:
    print("\n * Applying filter...")
    datasets = os.listdir(input_directory)
    for dataset in datasets:
        dataset_path = input_directory + "/" + dataset
        if os.path.isdir(dataset_path):
            data = pd.read_csv(dataset_path + "/dataset.csv")

        # Applies the filter to each data channel:
        for channel in range(14):
            data[f"Channel {channel+1}"] = filtfilt(b, a, data[f"Channel {channel+1}"])

        # Saves the data:
        output_dataset_directory = os.path.join(output_directory, dataset)
        if not os.path.exists(output_dataset_directory):
            os.makedirs(output_dataset_directory)

        output_file_path = os.path.join(output_dataset_directory, "dataset.csv")
        data.to_csv(output_file_path, index=False)

    print(" * Datasets correctly filtered :3")


# Function to apply Fourier Fast Transform to datasets:
def fourier_datasets(input_directory: str, output_directory: str):

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Applies FFT to each dataset:
    print("\n * Applying Fourier...")
    datasets = os.listdir(input_directory)
    for dataset in datasets:
        dataset_path = input_directory + "/" + dataset
        if os.path.isdir(dataset_path):
            data = pd.read_csv(dataset_path + "/dataset.csv")

            # Apply FFT to each data channel:
            for channel in range(14):
                channel_name = f"Channel {channel+1}"
                signal = data[channel_name].values
                fft_result = np.fft.fft(signal)
                data[channel_name] = np.abs(fft_result)

            # Saves the data with FFT applied:
            output_dataset_directory = os.path.join(output_directory, dataset)
            if not os.path.exists(output_dataset_directory):
                os.makedirs(output_dataset_directory)

            output_file_path = os.path.join(output_dataset_directory, "dataset.csv")
            data.to_csv(output_file_path, index=False)

    print(" * Datasets correctly Fouriered :3")


# Function to normalizate dataset:
def normalizate_datasets(input_directory, output_directory):

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Applies normalization to each dataset:
    print("\n * Applying normalization...")
    datasets = os.listdir(input_directory)
    for dataset in datasets:
        dataset_path = input_directory + "/" + dataset
        if os.path.isdir(dataset_path):
            data = pd.read_csv(dataset_path + "/dataset.csv")

        # Applies normalization to each data channel:
        mean = data.mean(axis=0, numeric_only=True)
        standev = data.std(axis=0, numeric_only=True)
        for channel in range(14):
            data[f"Channel {channel+1}"] = (data[f"Channel {channel+1}"] - mean[channel]) / standev[channel]

        # Saves the data:
        output_dataset_directory = os.path.join(output_directory, dataset)
        if not os.path.exists(output_dataset_directory):
            os.makedirs(output_dataset_directory)

        output_file_path = os.path.join(output_dataset_directory, "dataset.csv")
        data.to_csv(output_file_path, index=False)

    print(" * Datasets correctly normalizated :3")