import numpy as np
#Sigmoid activation
class Activation_Sigmoid: 

    #Forward pass
    def forward(self,inputs):
        #Save input and calculate the output
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    #Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)

    def backward(self, dvalues):
        #Need to modify the original variable,
        #Lets make a copy of the value first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0
