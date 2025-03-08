import math
from hiper_parameters import *

def der_sigmoid(x):
    return x*(1-x)

def sigmoid(x):
    return 1/(1+math.e**-x)

def softmax(z):
    import numpy as np
    z = np.array(z)  # Convertimos a un array de NumPy por seguridad
    z_max = np.max(z)  # Encontramos el máximo valor en `z`
    exp_z = np.exp(z - z_max)  # Restamos el máximo para estabilidad numérica
    return exp_z / np.sum(exp_z)

def gradient_sigmoid_cross_entropy(y_pred, y_true):
    return y_pred - y_true  

def gradient_sigmoid_errorsqr(expected,output):
    return (expected-output)*der_sigmoid(output)

#se debe dar los gradientes de la derecha y el h de una capa exacta
def gradient_hidden_sigmoid(gradients,weights,h):
    sum=0
    for w,g in zip(weights,gradients):
        sum+=w*g
    return sum*der_sigmoid(h)

def relu(x):
    return max(0, x)
    #return x if x>0 else alfa*x

def der_relu(x):
    return 1 if x > 0 else 0
    #return 1 if x > 0 else alfa

def gradient_hidden_relu(gradients,weights,h):
    sum=0
    for w,g in zip(weights,gradients):
        sum+=w*g
    return sum*der_relu(h)

def he_init(n_inputs):
    import random
    return [random.uniform(-1, 1) * math.sqrt(2 / n_inputs) for _ in range(n_inputs)]

def get_learning_rate(initial_lr, epoch,decay_rate):
    return initial_lr / (1 + decay_rate * epoch)