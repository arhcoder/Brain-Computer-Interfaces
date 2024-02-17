import os
import pandas as pd

#* RESAMPLING DATASETS TO BALANCE CLASSES *#
# Function to get the Imbalance Ratio for datasets:
def get_ir(dataset_path: str):

    # Datasets list:
    directory_content = os.listdir(dataset_path)

    # For each dataset:
    irs = list()
    for directory in directory_content:

        # Gets the dataset and counts the classes:
        dataset = pd.read_csv(dataset_path+"/"+directory+"/dataset.csv")
        class_counts = dataset["Expected Output"].value_counts()

        # Majority class:
        maj_class = class_counts.idxmax()
        maj_class_amount = class_counts.max()

        # Minority class:
        min_class = class_counts.idxmin()
        min_class_amount = class_counts.min()

        # Calculates the Imbalance Ratio (IR) for the dataset:
        ir = {"Dataset": directory, "IR": float(maj_class_amount / min_class_amount)}
        irs.append(ir)

    irs = sorted(irs, key=lambda x: x["Dataset"])
    return irs


# Function to crop timeseries on desbalanced datasets:
def crop_series(dataset: str, target_class: str, target_length: int):
    # Filter only for the target class:
    filtered_data = dataset[dataset["Expected Output"] == target_class]

    # Calculates how many records crop up to down:
    num_records_to_remove = len(filtered_data) - target_length

    # Identifies indexes to crop:
    indices_to_remove = filtered_data.head(num_records_to_remove).index #creo que esos indices no son los de dataset sino los de filtered_data y por eso solo balancea los que empiezan con muchas etiquetas Arbol

    # Crop records on the majoritary class:
    reduced_data = dataset.drop(indices_to_remove)
    return reduced_data


# Function to balance classes:
def balance_datasets(original_path: str, balanced_path: str, expected_size: int):

    # Gets the list of datasets:
    directory_content = os.listdir(original_path)

    # For each dataset:
    for directory in directory_content:
        dataset = pd.read_csv(f"{original_path}/{directory}/dataset.csv")
        class_counts = dataset["Expected Output"].value_counts()
        maj_class = class_counts.idxmax()

        # Gets the IR:
        maj_class_amount = class_counts[maj_class]
        min_class_amount = class_counts.min()
        ir = maj_class_amount / min_class_amount

        # If IR > 2 the class is balanced:
        if ir >= 2:
            # print(f"\n * Balanceando {directory}...")
            # Calculates the average length objetive:
            # average_length = sum(class_counts[class_counts.index != maj_class]) // (len(class_counts) - 1)
            average_length = expected_size

            # Crops the time series on majoritary classes:
            reduced_dataset = crop_series(dataset, maj_class, average_length)

            # Saves the balanced dataset on "balanced":
            os.makedirs(f"{balanced_path}/{directory}", exist_ok=True)
            reduced_dataset.to_csv(f"{balanced_path}/{directory}/dataset.csv", index=False)
            print(f" * {directory} balanced!")
        else:
            # If dataset is currently balanced, only copy the original:
            os.makedirs(f"{balanced_path}/{directory}", exist_ok=True)
            dataset.to_csv(f"{balanced_path}/{directory}/dataset.csv", index=False)


def drop_datasets(unusable_datasets: list, path: str):
    # Drops the datasets:
    for dataset_name in unusable_datasets:

        # Checks if path exists:
        dataset_path = os.path.join(path, dataset_name)
        if os.path.exists(dataset_path):
            try:
                os.remove(dataset_path+"/dataset.csv")
                os.rmdir(dataset_path)
                print(f" * {dataset_name} exterminated!")
            except OSError as e:
                print(f"Cannot eliminate [{dataset_name}].\nError: {e}")
        else:
            print(f"Dataset [{dataset_name}] does not exists :c")


def balance_exceptions(path: str):
    # Balanace an exception:
    dataset = pd.read_csv(f"{path}/dataset.csv")
    class_counts = dataset["Expected Output"].value_counts()
    maj_class = class_counts.idxmax()
    # print(f"\n * Balanceando {x}...")
    average_length = 1024

    # Crops majoritary classes series:
    reduced_dataset = crop_series(dataset, maj_class, average_length)

    # Saves the balanced dataset on "balanced":
    os.makedirs(path, exist_ok=True)
    reduced_dataset.to_csv(f"{path}/dataset.csv", index=False)
    dataset = path[path.rfind("/") + 1:]
    print(f" * {dataset} balanced!")