import os
import pandas as pd
import matplotlib.pyplot as plt

#* DATASET EXPLORATION *#
# Original dataset directory content:
dataset_path = "../Code/Datasets/original"
if os.path.exists(dataset_path):
    directory_content = os.listdir(dataset_path)
    print(f"Directory {dataset_path} content:")
    print(directory_content)
    print("Datasets amount:", len(directory_content))
else:
    print(f"Path {dataset_path} is incorrect! :c")

# Content of an specific dataset:
example_dataset = pd.read_csv(dataset_path+"/Alex1/dataset.csv")
print(example_dataset.head(5))

# Classes count on column "Expected Output":
class_counts = example_dataset["Expected Output"].value_counts()
print("Classes count:")
print(class_counts)

# Classes porportions on the dataset:
classes_proportions = class_counts / len(example_dataset)
plt.figure(figsize=(4, 4))
plt.pie(classes_proportions, labels=classes_proportions.index, autopct='%1.1f%%', startangle=140)
plt.title("Classes Proportions")
plt.axis("equal")
plt.show()