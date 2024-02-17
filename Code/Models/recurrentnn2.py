import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

# Define la arquitectura de la red neuronal:
class EEGClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EEGClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.feedforward = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.feedforward(out[:, -1, :])
        return out

base_path = "../DATA/original-balanced-filtered-fourier/Movement"
datasets = os.listdir(base_path)
print()

# Número de folds para K-Folds:
folds = 5
output_file = open("RecurrentNN2-metrics.txt", "w")
all_accuracy_values = []
all_recall_macro_values = []
all_f1_macro_values = []
all_precision_class_values = []
# Para cada SUJETO:
for dataset in datasets:
    # 1. Lee el dataset desde la dirección:
    dataset_path = os.path.join(base_path, dataset)

    # Verifica si es una carpeta y contiene un archivo "dataset.csv":
    if os.path.isdir(dataset_path) and "dataset.csv" in os.listdir(dataset_path):
        print(f"\nDATASET: {base_path}\nSUJETO: {dataset}\n")

        # 2. Ventaneo de datos:
        # Equivalente a 0.12 segundos:
        window_size = 30
        channels = 14
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

        # Convierte datos y etiquetas a tensores de PyTorch:
        X = torch.Tensor(X)
        y = torch.Tensor(y).long()

        # Crea un modelo de Red Neuronal:
        input_size = channels
        hidden_size = 64
        num_classes = 4
        model = EEGClassifier(input_size, hidden_size, num_classes)

        # Define la función de pérdida y el optimizador:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 3. Entrenar la Red Neuronal con K-fold:
        kfold = KFold(n_splits=folds, shuffle=True, random_state=11)
        fold_accuracy_values = []
        fold_recall_macro_values = []
        fold_f1_macro_values = []
        fold_precision_class_values = []
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Entrena la Red Neuronal:
            for epoch in tqdm(range(300)):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()

            # Evalúa el modelo en el conjunto de prueba:
            with torch.no_grad():
                model.eval()
                outputs = model(X_test)
                _, predicted = torch.max(outputs, 1)

                 # Calcula métricas para este fold:
                accuracy = accuracy_score(y_test, predicted)
                recall_macro = recall_score(y_test, predicted, average="macro")
                f1_macro = f1_score(y_test, predicted, average="macro")
                precision_class = precision_score(y_test, predicted, average=None)

                # Almacena las métricas para este fold:
                fold_accuracy_values.append(accuracy)
                fold_recall_macro_values.append(recall_macro)
                fold_f1_macro_values.append(f1_macro)
                fold_precision_class_values.append(precision_class)

                model.train()
        
        # Calcula el promedio de las métricas para este dataset:
        avg_accuracy = np.mean(fold_accuracy_values)
        avg_recall_macro = np.mean(fold_recall_macro_values)
        avg_f1_macro = np.mean(fold_f1_macro_values)
        avg_precision_class = np.mean(fold_precision_class_values, axis=0)

        # Almacena los promedios para este dataset:
        all_accuracy_values.append(avg_accuracy)
        all_recall_macro_values.append(avg_recall_macro)
        all_f1_macro_values.append(avg_f1_macro)
        all_precision_class_values.append(avg_precision_class)

# Imprime y escribe en el archivo los resultados globales:
# Calcula promedio global:
avg_global_accuracy = np.mean(all_accuracy_values)
avg_global_recall = np.mean(all_recall_macro_values)
avg_global_f1 = np.mean(all_f1_macro_values)
avg_precision_class_values = np.mean(all_precision_class_values, axis=0)

output_file.write("\n-----------------------------------\n")
output_file.write("\nRRN LSTM: INTENCIÓN DE MOVIMIENTO")
output_file.write("\nPromedio Global de Métricas")
output_file.write("\n\n-----------------------------------\n")
output_file.write(f"\n* Accuracy: {avg_global_accuracy:.4f}\n")
output_file.write(f"* Recall Macro: {avg_global_recall:.4f}\n")
output_file.write(f"* F1 Macro: {avg_global_f1:.4f}\n")
output_file.write("* Precision:\n")
class_names = ["MouseUp", "MouseDown", "MouseLeft", "MouseRight"]
for class_label, avg_precision in enumerate(avg_precision_class_values):
    output_file.write(f"  * Clase \"{class_names[class_label]}\": {avg_precision:.4f}\n")
output_file.write("\n-----------------------------------")
# output_file.write(f"\n* Accuracy más alto: {highest_accuracy:.4f}\n")
# output_file.write(f"* Accuracy más bajo: {lowest_accuracy:.4f}\n")
# output_file.write("\n---------------------------------\n")

output_file.close()