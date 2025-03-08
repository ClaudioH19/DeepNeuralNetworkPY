from funciones import *

class neuron:    
    def __init__(self,cant_neur_anterior):
        self.weights = he_init(cant_neur_anterior)
        self.z=0
        self.h=0
        self.gradient=0
        self.bias= 0
        
    def calc_z(self, inputs):
        self.z = sum(float(w) * float(i) for w, i in zip(self.weights, inputs)) + float(self.bias)
        return self.z

       

class network:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes)):
            neurons = []
            cant_neur_anterior = layer_sizes[i - 1] if i > 0 else layer_sizes[i]  # Tamaño de entrada si es la primera capa
            for _ in range(layer_sizes[i]):  
                neurons.append(neuron(cant_neur_anterior))  # Se ajusta el tamaño de los pesos
            self.layers.append(neurons)

def forward(network, inputs):
    for idx_layer, layer in enumerate(network.layers):
        for neuron in layer:
            ### Calcular Z ###
            if idx_layer == 0:
                neuron.calc_z(inputs)  # Primera capa usa inputs
            else:
                neuron.calc_z([n.h for n in network.layers[idx_layer - 1]])  # Usa salida de la capa anterior

        ### Activación ###
        if idx_layer == len(network.layers) - 1:  # Última capa
            for idx_neuron, neuron in enumerate(layer):
                neuron.h= sigmoid(neuron.z)
        else:  # Capas ocultas
            for neuron in layer:
                neuron.h = relu(neuron.z)  # Aplicar ReLU



def backward(network,expected):
    idx_layer=len(network.layers)-1
    while (idx_layer>=0):
        layer=network.layers[idx_layer]
        for idx_neuron,neuron in enumerate(layer):
            ### calcular gradiente si es ultimo ###
            if idx_layer == len(network.layers)-1:
                neuron.gradient=gradient_sigmoid_cross_entropy(neuron.h,expected) 
            else:
                ### calcular gradiente para una hidden ###
                gradientes_der= [neuron.gradient for neuron in network.layers[idx_layer+1]]
                pesos_der = [neuron.weights[idx_neuron] for neuron in network.layers[idx_layer+1]]
                neuron.gradient=gradient_hidden_relu(gradientes_der,pesos_der,neuron.h)
        idx_layer-=1



def update(network,inputs,learning_rate,lambda_l2):
    idx_layer=len(network.layers)-1
    while (idx_layer>=0):
        layer=network.layers[idx_layer]
        for idx_neuron,neuron in enumerate(layer):
            neuron.bias-=neuron.gradient*learning_rate
            # Actualización de los pesos con L2 (sumando la regularización)
            for idx, w in enumerate(neuron.weights):
                # Gradiente regular más penalización L2
                l2_penalty = lambda_l2 * w  # L2
                if idx_layer == 0:
                    neuron.weights[idx] -= learning_rate * neuron.gradient * inputs[idx] + l2_penalty
                else:
                    neuron.weights[idx] -= learning_rate * neuron.gradient * network.layers[idx_layer - 1][idx].h + l2_penalty
        
        idx_layer-=1

            