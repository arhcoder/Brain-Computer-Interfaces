import os
import pandas as pd

#* SEPARATING DATASETS IMPORTANT INFORMATION *#
# Function to:
# 1. Remove "Nada" label data.
# 2. Replace targets with numbers.
# 3. Separate datasets in:
#    a. Concept Evocation.
#    b. Movement Intention.

# Function to process and reorganize datasets:
def separate_datasets(input_directory, output_directory):

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Process and reorganize each dataset:
    print("\n * Separating datasets...")
    datasets = os.listdir(input_directory)
    for dataset in datasets:
        dataset_path = os.path.join(input_directory, dataset)
        if os.path.isdir(dataset_path):
            data = pd.read_csv(os.path.join(dataset_path, "dataset.csv"))

            # Step 1: Remove rows where "Expected Output" is "Nada":
            data = data[data["Expected Output"] != "Nada"]

            # Step 2: Determine the destination folder based on the name of the dataset folder:
            if dataset.endswith("1"):
                destination_folder = "Concept"
            elif dataset.endswith("2"):
                destination_folder = "Movement"
            else:
                destination_folder = "Other"

            # Step 3: Change "Expected Output" labels with numbers:
            if destination_folder == "Concept":
                label_mapping = {
                    "Arbol": 0,
                    "Cuaderno": 1,
                    "Computadora": 2,
                    "Perro": 3
                }
            elif destination_folder == "Movement":
                label_mapping = {
                    "MouseUp": 0,
                    "MouseDown": 1,
                    "MouseLeft": 2,
                    "MouseRight": 3
                }
            else:
                pass
            data["Expected Output"] = data["Expected Output"].map(label_mapping)
            data = data.rename(columns={"Expected Output": "Target"})

            # Save the processed data to the destination folder:
            output_dataset_directory = os.path.join(output_directory, destination_folder, dataset)
            if not os.path.exists(output_dataset_directory):
                os.makedirs(output_dataset_directory)

            output_file_path = os.path.join(output_dataset_directory, "dataset.csv")
            data.to_csv(output_file_path, index=False)

    print(" * Datasets correctly separated :3")