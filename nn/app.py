# coding=utf-8

import numpy as np
import pandas as pd
import pickle
from nn.layer.layer import Layer_Dense
from nn.activation.activation_function import Activation_Sigmoid, Activation_ReLU
from nn.loss.loss_function import Loss_BinaryCrossEntropy
from nn.optimizer.optimizer import Optimizer_Adam

class Model:
    def __init__(self, layer1, layer2, activation1, activation2):
        self.layer1 = layer1
        self.layer2 = layer2
        self.activation1 = activation1
        self.activation2 = activation2

def run(x):
    model_in = open('/Users/luleano/Documents/proyecto_modular/hssad_NN/model.pickle','rb')
    model = pickle.load(model_in)
    print(predict(x,model))
    

def save_parameters(self, path):
    #Open a file in the binary-write code
    #and save parameters to it 
    #Guardar los objetos weights , bias 
    with open(path, 'wb') as f:
        pickle.dump(self.parameters, f)
#Loads the weights and updates a model instance with them
def load_parameters(self, path):
    #Open file in the binary-read mode,
    #load weights and bias
    with open(path, 'rb') as f:
        self.set_parameters(pickle.load(f))

def predict(x, model):
    model.layer1.forward(x)

    model.activation1.forward(model.layer1.output)

    model.layer2.forward(model.activation1.output)

    model.activation2.forward(model.layer2.output)
    result= (model.activation2.output > 0.5) *1
    return result[0][0]

def train():
    #Getting X and y values from the file
    df = pd.read_csv("/Users/luleano/Documents/proyecto_modular/hssad_NN/nn/resources/dataset.csv")
    
    #Getting the inputs and y values from the csv
    #Integer position-based
    X = df.iloc[:, 1:21].to_numpy()
    #Label bassed 
    y = df.loc[:, ['y']].to_numpy() 

    x_training, x_test = X[:794,:], X[794:,:]
    y_training, y_test = y[:794,:], y[794:,:]
    
    #Training the network 
    layer1 = Layer_Dense(20,20, weight_regularizer_l2=5e-4,
                            bias_regularizer_l2=5e-4)
    activation1 = Activation_ReLU()
    layer2 = Layer_Dense(20,1)
    activation2 = Activation_Sigmoid()
    loss_function = Loss_BinaryCrossEntropy()
    # Create optimizer
    optimizer = Optimizer_Adam(decay=5e-7)

    for epoch in range(1001):
        layer1.forward(x_training)
        
        activation1.forward(layer1.output)

        layer2.forward(activation1.output)

        activation2.forward(layer2.output)

        data_loss = loss_function.calculate(activation2.output, y_training)
         # Calculate regularization penalty
        regularization_loss = \
            loss_function.regularization_loss(layer1) + \
            loss_function.regularization_loss(layer2)

        # Calculate overall loss
        loss = data_loss + regularization_loss
        
        #Calculate accuracy from output of activation2 and targets
        predictions = (activation2.output > 0.5) * 1
        accuracy = np.mean(predictions==y_training)
       
        if not epoch % 100:
            print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, '+
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')

        #Backward pass
        loss_function.backward(activation2.output, y_training)
        activation2.backward(loss_function.dinputs)
        layer2.backward(activation2.dinputs)
        activation1.backward(layer2.dinputs)
        layer1.backward(activation1.dinputs)

        #Update weights and biases
        #Calling optimizer
        optimizer.pre_update_params()
        optimizer.update_params(layer1)
        optimizer.update_params(layer2)
        optimizer.post_update_params()
    
    #Validate Model
    layer1.forward(x_test)

    activation1.forward(layer1.output)

    layer2.forward(activation1.output)

    activation2.forward(layer2.output)

    loss = loss_function.calculate(activation2.output, y_test)

    predictions = (activation2.output > 0.5) *1
    accuracy = np.mean(predictions==y_test)

    print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')

    #Saving the model to future use predicting
    classification_model= Model(layer1,layer2, activation1, activation2)
    model_out = open('model.pickle','wb')
    pickle.dump(classification_model, model_out)
    model_out.close()



        

    
        

    

    
     
