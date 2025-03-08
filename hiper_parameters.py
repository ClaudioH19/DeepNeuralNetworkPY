inputs=16
capas=[inputs,8,4,2,1]
epochs=100

lambda_l2=0.000007
alfa=0.01
initial_lr = 0.5
decay_rate=0.9

if __name__ == "__main__":
    from entrenamiento import training
    training(epochs,initial_lr,lambda_l2,decay_rate)