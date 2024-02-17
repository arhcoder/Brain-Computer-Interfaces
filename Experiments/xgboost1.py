import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, KFold

base_path = "../DATA/original-balanced-filtered-fourier-normalizated/Concept"
datasets = os.listdir(base_path)
print()

# Número de folds para K-Folds:
folds = 5
all_metrics = {}
output_file = open("XGBoost1-metrics.txt", "w")

# Para cada SUJETO:
for dataset in datasets:
    # 1. Lee el dataset desde la dirección:
    dataset_path = os.path.join(base_path, dataset)
            
    # Verifica si es una carpeta y contiene un archivo "dataset.csv":
    if os.path.isdir(dataset_path) and "dataset.csv" in os.listdir(dataset_path):
        print(f"DATASET: {base_path}\nSUJETO: {dataset}\n")

        # 2. Ventaneo de datos:
        # Equivalente a 0.12 segundos:
        window_size = 30
        X, y = [], []
        data_path = os.path.join(dataset_path, "dataset.csv")
        data = pd.read_csv(data_path)

        for i in range(0, len(data) - window_size + 1, window_size):
            window = data.iloc[i:i + window_size, :-1]
            label = data.iloc[i + window_size - 1, -1]
            X.append(window.values)
            y.append(label)
        X = np.array(X)
        y = np.array(y)

        # 3. Entrena el modelo XGBoost:
        model = XGBClassifier()
        kfold = KFold(n_splits=folds, shuffle=True, random_state=11)
        scores = cross_validate(model, X.reshape(X.shape[0], -1), y, cv=kfold, scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"])

        # 4. Calcula métricas y almacena en el diccionario:
        avg_metrics = {"accuracy": np.mean(scores["test_accuracy"])}
        avg_metrics["recall"] = scores["test_recall_macro"]
        avg_metrics["precision"] = scores["test_precision_macro"]
        avg_metrics["f1"] = scores["test_f1_macro"]

        all_metrics[dataset] = avg_metrics

        # 5. Imprime resultados en consola y escribe en un log:
        output_file.write(f"{dataset}:\n")
        output_file.write(f"* Accuracy: {avg_metrics['accuracy']:.4f}\n")
        output_file.write("* Recall:\n")
        for class_label, class_recall in enumerate(avg_metrics["recall"][:-1]):
            output_file.write(f"    * Clase {class_label}: {class_recall:.4f}\n")
        output_file.write("* F1:\n")
        for class_label, class_f1 in enumerate(avg_metrics["f1"][:-1]):
            output_file.write(f"    * Clase {class_label}: {class_f1:.4f}\n")
        output_file.write("* Precision:\n")
        for class_label, class_precision in enumerate(avg_metrics["precision"][:-1]):
            output_file.write(f"    * Clase {class_label}: {class_precision:.4f}\n")
output_file.write("\n")

# Calcula y almacenar promedio global y accuracies más alto y más bajo:
all_accuracy_values = []
all_recall_macro_values = []
all_f1_macro_values = []
all_precision_class_values = [[] for _ in range(4)]

highest_accuracy = 0.0
lowest_accuracy = 1.0

for dataset, metrics in all_metrics.items():
    # Guarda accuracies para calcular el máximo y mínimo:
    accuracy_value = metrics["accuracy"]
    all_accuracy_values.append(accuracy_value)

    # Actualiza accuracies más alto y más bajo:
    highest_accuracy = max(highest_accuracy, accuracy_value)
    lowest_accuracy = min(lowest_accuracy, accuracy_value)

    # Guarda el recall y f1 para calcular promedio global:
    all_recall_macro_values.extend(metrics["recall"])
    all_f1_macro_values.extend(metrics["f1"])

    # Guarda precision por clase:
    for class_label, class_precision in enumerate(metrics["precision"][:-1]):
        all_precision_class_values[class_label].append(class_precision)
    # Promedio por clase:
    avg_precision_class_values = [np.mean(class_values) for class_values in all_precision_class_values]

# Calcula promedio global:
avg_global_accuracy = np.mean(all_accuracy_values)
avg_global_recall = np.mean(all_recall_macro_values)
avg_global_f1 = np.mean(all_f1_macro_values)

# Imprime y escribe en el archivo los resultados globales:
output_file.write("\n---------------------------------\n")
output_file.write("\nXGBOOST: EVOCACIÓN DE CONCEPTO")
output_file.write("\nPromedio Global de Métricas")
output_file.write("\n\n---------------------------------\n")
output_file.write(f"\n* Accuracy: {avg_global_accuracy:.4f}\n")
output_file.write(f"* Recall Macro: {avg_global_recall:.4f}\n")
output_file.write(f"* F1 Macro: {avg_global_f1:.4f}\n")
output_file.write("* Precision:\n")
class_names = ["Árbol", "Cuaderno", "Computadora", "Perro"]
for class_label, avg_precision in enumerate(avg_precision_class_values):
    output_file.write(f"  * Clase \"{class_names[class_label]}\": {avg_precision:.4f}\n")
output_file.write("\n---------------------------------\n")
output_file.write(f"\n* Accuracy más alto: {highest_accuracy:.4f}\n")
output_file.write(f"* Accuracy más bajo: {lowest_accuracy:.4f}\n")
output_file.write("\n---------------------------------\n")

output_file.close()