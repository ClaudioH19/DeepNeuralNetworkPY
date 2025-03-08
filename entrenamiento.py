from Red import *
from app import save_model
import csv
from funciones import *
import numpy as np

##############
###__init__###
##############
nombre_archivo = "diabetes_dataset.csv"
from hiper_parameters import *
network = network(capas)

# Función para normalizar las entradas usando Min-Max Scaling
def normalize_minmax(data, min_vals=None, max_vals=None):
    data = np.array(data)
    if min_vals is None:
        min_vals = np.min(data, axis=0)
    if max_vals is None:
        max_vals = np.max(data, axis=0)
    
    return (data - min_vals) / (max_vals - min_vals), min_vals, max_vals

# Función para equilibrar los datos antes de entrenar
def balance_data():
    class_0 = []
    class_1 = []
    
    # Cargar los datos y dividir por clase
    with open(nombre_archivo, newline='', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Saltar encabezado
        for row in reader:
            outcome = int(row[-1])
            if outcome == 0:
                class_0.append(row)
            else:
                class_1.append(row)

    # Seleccionar 1000 muestras de cada clase
    class_0 = class_0[:1000]
    class_1 = class_1[:1000]
    
    # Combinar las muestras balanceadas
    balanced_data = class_0 + class_1
    
    # Mezclar los datos para evitar que queden secuenciales
    np.random.shuffle(balanced_data)
    
    return balanced_data, class_0 + class_1  # Devolver conjunto balanceado y el resto para pruebas

def training(epochs, initial_lr, lambda_l2, decay_rate):
    hits = 0
    misses = 0

    # Cargar y equilibrar los datos
    balanced_data, test_data = balance_data()

    # Leer los datos para calcular min y max antes del entrenamiento
    all_data = np.array([list(map(float, row[:-1])) for row in balanced_data])
    min_vals = np.min(all_data, axis=0)
    max_vals = np.max(all_data, axis=0)

    for i in range(epochs):
        for idx, row in enumerate(balanced_data):
            lr = get_learning_rate(initial_lr, idx, decay_rate)
            raw_inputs = list(map(float, row[:-1]))
            inputs, _, _ = normalize_minmax([raw_inputs], min_vals, max_vals)  # Normalizar antes de pasar a la red
            inputs = inputs[0]  # Convertir de array 2D a 1D

            outcome = int(row[-1])

            forward(network, inputs)
            backward(network, outcome)
            update(network, inputs, lr, lambda_l2)

            output = network.layers[len(network.layers)-1][0].h
            output_quality = 1 if output > 0.5 else 0  # Seleccionar la predicción más alta
            if outcome == output_quality:
                hits += 1
            else:
                misses += 1
            print(f"Acierto: {hits / (hits + misses) * 100}% Epoch: {i}")

    hits = 0
    misses = 0
    # Crear un archivo de resultados para la prueba
    with open('resultados_prueba.txt', 'w') as result_file:
        result_file.write("Esperado\tPredicción de la Red\n")  # Encabezado del archivo

        # Probar el modelo con el resto de los datos
        for row in test_data:
            raw_inputs = list(map(float, row[:-1]))
            inputs, _, _ = normalize_minmax([raw_inputs], min_vals, max_vals)
            inputs = inputs[0]  # Convertir de array 2D a 1D

            outcome = int(row[-1])

            forward(network, inputs)

            # Cálculo de la predicción
            output = network.layers[len(network.layers)-1][0].h
            output_quality = 1 if output > 0.5 else 0

            # Escribir en el archivo de resultados
            result_file.write(f"{outcome}\t{output_quality}\n")
            
            if outcome == output_quality:
                hits += 1
            else:
                misses += 1
            print(f"Esperado {outcome} vs RED: {output_quality}")
            print(f"Acierto en PRUEBA: {hits / (hits + misses) * 100}%")

    if hits / (hits + misses) * 100 > 40:
        save_model(network)
