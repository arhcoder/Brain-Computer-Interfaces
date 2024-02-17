
from resampling import balance_datasets
from resampling import drop_datasets
from resampling import balance_exceptions

from transformation import filter_datasets
from transformation import fourier_datasets
from transformation import normalizate_datasets

from separation import separate_datasets
import os


#/ Direction of original dataset:
dataset_path = "./Datasets/original"


#* RESAMPLING *#
#* -----------------------------------------------------------------------------------
#? Balance datasets:
balanced_path = "./Datasets/original-balanced"
balance_datasets(dataset_path, balanced_path, expected_size=1024)

#? Drops unusable datasets:
unusable_datasets = ["pep2", "Oscar2", "Hiram2", "Abraham1", "Abraham2"]
drop_datasets(unusable_datasets, balanced_path)

#? Balance individual dataset exception:
exception_dataset = balanced_path+"/Andy1"
balance_exceptions(exception_dataset)
#* -----------------------------------------------------------------------------------


#* TRANSFORMATION *#
#* -----------------------------------------------------------------------------------
#? Filtering datasets with pass band Butterworth filter:
filtered_path = "./Datasets/original-balanced-filtered"
filter_datasets(balanced_path, filtered_path, lowcut=12, highcut=50, frecuency=256)

#? Applying Fourier Transformation to datasets:
fourier_path = "./Datasets/original-balanced-filtered-fourier"
fourier_datasets(filtered_path, fourier_path)

#? Applying Normalization:
normalizated_path = "./Datasets/original-balanced-filtered-fourier-normalizated"
normalizate_datasets(fourier_path, normalizated_path)

#? Applying Normalization:
xnormalizated_path = "./Datasets/original-balanced-filtered-normalizated"
normalizate_datasets(filtered_path, xnormalizated_path)
#* -----------------------------------------------------------------------------------


#* SEPARATION *#
#* -----------------------------------------------------------------------------------
#? Separation for all the datasets:
base_path = "./Datasets/"
datasets = os.listdir(base_path)
for dataset in datasets:
    xinput_directory = base_path+dataset
    xoutput_directory = "./DATA/"+dataset
    separate_datasets(xinput_directory, xoutput_directory)

# Special Datasets:
# Original => Fourier:
out_path = "./Datasets/original-fourier"
fourier_datasets(dataset_path, out_path)
separate_datasets(out_path, "./DATA/original-fourier")

# Original => Fourier => Normalizated:
normalizate_datasets(out_path, "./Datasets/original-fourier-normalizated")
out_path = "./Datasets/original-fourier-normalizated"
separate_datasets(out_path, "./DATA/original-fourier-normalizated")

#* -----------------------------------------------------------------------------------