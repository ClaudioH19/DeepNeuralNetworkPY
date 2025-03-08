from flask import Flask, render_template, request, jsonify
import numpy as np
import json

def save_model(network, filename="modelo.json"):
    model_data = []
    for layer in network.layers:
        layer_data = []
        for neuron in layer:
            neuron_data = {
                "weights": neuron.weights.tolist(),
                "bias": neuron.bias
            }
            layer_data.append(neuron_data)
        model_data.append(layer_data)
    
    with open(filename, "w") as f:
        json.dump(model_data, f)
    print(f"Modelo guardado en {filename}")

def load_model(network, filename="modelo.json"):
    with open(filename, "r") as f:
        model_data = json.load(f)
    
    for layer, layer_data in zip(network.layers, model_data):
        for neuron, neuron_data in zip(layer, layer_data):
            neuron.weights = np.array(neuron_data["weights"])
            neuron.bias = neuron_data["bias"]

def normalize_minmax(data, min_vals, max_vals):
    data = np.array(data)
    return (data - min_vals) / (max_vals - min_vals)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        from Red import forward, network
        from hiper_parameters import capas
        net = network(capas)
        load_model(net)

        data = request.form
        inputs = [
            float(data['age']), float(data['pregnancies']), float(data['bmi']), float(data['glucose']),
            float(data['blood_pressure']), float(data['hba1c']), float(data['ldl']), float(data['hdl']),
            float(data['triglycerides']), float(data['waist_circumference']), float(data['hip_circumference']),
            float(data['whr']), float(data['family_history']), float(data['diet_type']), 
            float(data['hypertension']), float(data['medication_use'])
        ]

        min_vals = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        max_vals = np.array([100, 50, 50, 200, 150, 10, 250, 100, 300, 200, 200, 1, 1, 1, 1, 1])

        normalized_inputs = normalize_minmax([inputs], min_vals, max_vals)[0]
        forward(net, normalized_inputs)
        
        result = 1 if net.layers[-1][0].h > 0.5 else 0

        # Obtener datos de la red para la visualización
        network_data = []
        for layer in net.layers:
            layer_info = []
            for neuron in layer:
                layer_info.append({"weights": neuron.weights.tolist(), "activation": neuron.h})
            network_data.append(layer_info)
        
        return render_template('index.html', result=result, network_data=json.dumps(network_data))
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
