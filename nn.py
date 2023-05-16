import torch as tch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from neural_network import NeuralNetwork

modelo_cargado = NeuralNetwork()

ruta_modelo = './modelo.pth'
modelo_cargado.load_state_dict(tch.load(ruta_modelo))

def pruebas(df):
    #separamos predictores de etiquetas
    X_test = df.iloc[:, :-1].astype('float32').to_numpy()
    Y_test = df.iloc[:, -1]

    # Convertir a numerico
    label_encoder = {label: i for i, label in enumerate(set(Y_test))}
    Y_test = Y_test.map(label_encoder).astype('float32').to_numpy().reshape(-1, 1)

    # Convertir a Tensors
    X_test = tch.tensor(X_test, dtype=tch.float32)
    Y_test = tch.tensor(Y_test, dtype=tch.float32)

    # imprime para verificar
    #print("X shape:", X_test.shape)
    #print("Y shape:", Y_test.shape)

    salida = modelo_cargado(X_test)
    probabilidades = F.softmax(salida, dim=1)
    
    # print(probabilidades)
    
    #print("DF Shape:",probabilidades.shape)

    # Obtener las etiquetas predichas
    Y_pred = tch.argmax(probabilidades, dim=1)

    # Convertir etiquetas a NumPy arrays
    Y_test = Y_test.squeeze().numpy()
    Y_pred = Y_pred.squeeze().numpy()

    #print("Etiquetas: ",Y_pred)
    #print("numero de Etiquetas: ",Y_pred.shape)

    # Obtener las métricas
    report = classification_report(Y_test, Y_pred)
    matrix = confusion_matrix(Y_test, Y_pred)

    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", matrix)

    return probabilidades, Y_pred

def procesar_data(csv_path):
    df = pd.read_csv(csv_path, sep=',')
    probabilidades, Y_pred=pruebas(df)
    # Obtener etiquetas únicas
    unique_labels, counts = np.unique(Y_pred, return_counts=True)

    # Imprimir etiquetas y conteos
    for label, count in zip(unique_labels, counts):
        print("Etiqueta {}: {} veces".format(label+1, count))

    # Obtener la etiqueta que tiene mayor conteo
    max_count_index = np.argmax(counts)
    max_count_label = unique_labels[max_count_index]

    # Imprimir la etiqueta que tiene mayor conteo
    print("se clasifica como cancion {}: se repite en {} muestras".format(max_count_label+1, counts[max_count_index]))
    # para sacar un porcentaje de similaridad por la cancion 

    porcentaje=counts[max_count_index]/len(Y_pred)*100

    print("Porcentaje ", porcentaje)
    return porcentaje

