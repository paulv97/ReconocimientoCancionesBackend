import librosa
import librosa.display
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer

def sacar_espectrograma(audio, etiqueta):
    y, sr = librosa.load(audio) # carga el audio y la tasa de muestreo, y: longitud de audio forma de onda , sr: tasa de muestreo
    
    # Tomar los primeros 60 segundos de la señal de audio
    objetivo_duration = 60  #se establece duracion de 60 segundos
    objetivo_length = sr * objetivo_duration #longitud objetivo en muestras
    if len(y) > objetivo_length: # confirma si lalonfitud es mayor recorta o rellena de ceros si falta
        y = y[:objetivo_length]
    else:
        y = librosa.util.fix_length(y, objetivo_length)
    
    # Definicion de parametros  espectrograma
    n_fft = 2048                #tamaño de la ventana de transformada de Fourier
    hop_length = 512            #la cantidad de muestras entre ventanas sucesivas
    n_mels = 128                #es el número de bandas mel en el espectrograma
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)   #calcula el espectrograma de mel
    S_dB = librosa.power_to_db(S, ref=np.max)                 #power_to_db se utiliza para convertir la amplitud del espectrograma a una escala de decibeles (dB).

    # Crear matriz de espectrograma con etiqueta
    espectrograma = pd.DataFrame(S_dB.T)  #La transposición es necesaria para que cada fila represente un instante de tiempo y cada columna represente una banda mel.
    espectrograma["label"] = etiqueta     # se coloca la etiqueta

    return  espectrograma

def normalizar(cancion):
    # Seleccionar todas las columnas excepto la última
    X = cancion.iloc[:, :-1]
    # Escalador Min-Max
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Normalizar los datos
    X_norm = scaler.fit_transform(X)
    # Concatenar las columnas normalizadas con la última columna
    cancion_norm = np.concatenate((X_norm, cancion.iloc[:, -1].values.reshape(-1, 1)), axis=1)
    return cancion_norm


#binarizar
def binarizar(cancion, umbral):
    cancion=pd.DataFrame(cancion)
    # Seleccionar todas las columnas excepto la última
    X = cancion.iloc[:, :-1].astype('float')
    # Binarizar los datos utilizando un umbral específico
    binarizer = Binarizer(threshold=umbral)
    X_bin = binarizer.fit_transform(X)
    # Concatenar las columnas binarizadas con la última columna
    cancion_bin = np.concatenate((X_bin, cancion.iloc[:, -1].values.reshape(-1, 1)), axis=1)
    return cancion_bin

def obtener_dataframe(data):
    return pd.DataFrame(data)