import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

#* DATASET VISUALIZATION *#
base_path = "../Code/Datasets"


#? Original Dataset:
dataset = pd.read_csv(base_path+"/original/Alex1/dataset.csv")
serieArbol = dataset.loc[dataset["Expected Output"] == "Arbol"]
seconds = np.arange(len(serieArbol)) / 256
plt.figure(figsize=(20, 10))
plt.plot(seconds, serieArbol["Channel 3"])
plt.xticks(ha="right")
plt.suptitle("Time Serie on EEG Channel 3 from \"profiles/Alex1\"")
plt.title("Original Dataset")
plt.xlabel("Seconds")
plt.ylabel("Hz Intensity")
plt.show()


#? Balanced Dataset:
dataset = pd.read_csv(base_path+"/original-balanced/Alex1/dataset.csv")
serieArbol = dataset.loc[dataset["Expected Output"] == "Arbol"]
seconds = np.arange(len(serieArbol)) / 256
plt.figure(figsize=(20, 10))
plt.plot(seconds, serieArbol["Channel 3"])
plt.xticks(ha="right")
plt.suptitle("Time Serie on EEG Channel 3 from \"balanced/Alex1\"")
plt.title("Balanced Dataset")
plt.xlabel("Seconds")
plt.ylabel("Hz Intensity")
plt.show()


#? Filtered Dataset:
dataset = pd.read_csv(base_path+"/original-balanced-filtered/Alex1/dataset.csv")
serieArbol = dataset.loc[dataset["Expected Output"] == "Arbol"]
seconds = np.arange(len(serieArbol)) / 256
plt.figure(figsize=(20, 10))
plt.plot(seconds, serieArbol["Channel 3"])
plt.xticks(ha="right")
plt.suptitle("Time Serie on EEG Channel 3 from \"filtered/Alex1\"")
plt.title("Filtered Dataset")
plt.xlabel("Seconds")
plt.ylabel("Hz")
plt.show()


#? Fourier Dataset:
dataset = pd.read_csv(base_path+"/original-balanced-filtered-fourier/Alex1/dataset.csv")
serieArbol = dataset.loc[dataset["Expected Output"] == "Arbol"]
seconds = np.arange(len(serieArbol)) / 256
plt.figure(figsize=(20, 10))
plt.plot(seconds, serieArbol["Channel 3"])
plt.xticks(ha="right")
plt.suptitle("Time Serie on EEG Channel 3 from \"filtered/Alex1\"")
plt.title("Fourier Dataset")
plt.xlabel("Seconds")
plt.ylabel("Hz")
plt.show()


#? Normalizated Dataset:
dataset = pd.read_csv(base_path+"/original-balanced-filtered-fourier-normalizated/Alex1/dataset.csv")
serieArbol = dataset.loc[dataset["Expected Output"] == "Arbol"]
seconds = np.arange(len(serieArbol)) / 256
plt.figure(figsize=(20, 10))
plt.plot(seconds, serieArbol["Channel 3"])
plt.xticks(ha="right")
plt.suptitle("Time Serie on EEG Channel 3 from \"normalizated/Alex1\"")
plt.title("Normalizated Dataset")
plt.xlabel("Seconds")
plt.ylabel("Normalizated Hz")
plt.show()


#? FINAL DATASET [ALL CHANNELS]:
dataset = pd.read_csv(base_path+"/original-balanced-filtered-fourier-normalizated/Alex1/dataset.csv")
serieArbol = dataset.loc[dataset["Expected Output"] == "Arbol"]
seconds = np.arange(len(serieArbol)) / 256

# Plots all the 14 channels:
plt.figure(figsize=(20, 10))
plt.xticks(ha="right")
plt.suptitle("Time Serie on EEG Channels from \"normalizated/Alex1\"")
plt.title("FINAL DATASET")
plt.xlabel("Seconds")
plt.ylabel("Normalizated Hz")
for channel in range(13):
    plt.plot(seconds, serieArbol[f"Channel {channel+1}"])
plt.show()