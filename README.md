# HEALTHCARE SUPPORT SYSTEM AGAINST DEPRESSION - NEURAL NETWORK
<img src="https://github.com/luisUrielLeano/HSSAD_REACT/blob/main/public/img/svg/logo.svg" width="250" height="250" />

Health Care Support System Against Depression is web platform designed to help common people to predict possible depression cases by
answering a specialized survey based on the CES-D scale; offering a reliable tool for accurate detection and prediction with Artificial Intelligence help using Neuronal Networks.
In this way, people can be aware of their mental health and seek professional help if necessary.

At the same time the project looks for collecting crucial data that could be useful for other projects and research studies. 

This project works in conjuntion with [HSSAD_REACT](https://github.com/luisUrielLeano/HSSAD_REACT) and [HSSAD_NODE](https://github.com/luisUrielLeano/HSSAD_NODE)

## General architecture of the project
<img src="https://github.com/luisUrielLeano/HSSAD_REACT/blob/main/public/img/generalArchitecture.png" />

## Neural Network Module 
The Neural Network was created from scratch using as reference the book Neural Networks from Scratch (NNFS) by Harrison Kinsley & Daniel Kukiela. https://nnfs.io

### General Architecture

The general architectura  for the Neural Network implemented in this project is the following:
* Input layer which contains 20 neurons; each one represents the answer to each question in the questionnaire, whose value could be a whole number between 1-4.

* Hidden layer with the same number of neurons, whose activation function is ReLu.

* Output layer with only one neuron whose activation function is Sigmoid. The output value could be 0 or 1.

<img src="https://github.com/luisUrielLeano/HSSAD_REACT/blob/main/public/img/nnGeneralArchitecture.png" width="600" height="350"/>

In the hidden layer each neuron will receive 20 values as input, that will be multiplied by their respective weights and then it will be added
each of these multiplications. Finally, it will be added a bias to previous summation and the result will be passed as input to the ReLu activation function.

If the input value is greater than 0 it will return the value itself, otherwise it will return 0.

<img src="https://github.com/luisUrielLeano/HSSAD_REACT/blob/main/public/img/hiddenLayer.png" width="400" height="200"/>

In the output layer each neuron will receive 20 values as input as well, that will be multiplied by their respective weights and then it will be added
each of these multiplications. Finally, it will be added a bias to previous summation and the result will be passed as input to the Sigmoid activation function,
that will return a value very close to 1 or a value very close to 0.

This is very useful to reduce the error and update the weights, but in the prediction phase is necessary to apply some validation where if the value is greater
than .5 return 1 , otherwise return 0. 

<img src="https://github.com/luisUrielLeano/HSSAD_REACT/blob/main/public/img/outputLayer.png" width="400" height="200"/>

## Dependencies
All code is written using Python 3.
* **Numpy** *1.20.3* for numerical calculation and handling of matrices.
* **Pandas** *1.2.4* for datasets construction and handling.
## Installation
Using the Conda package, dependency and environment management.

Create a environment from the `environment.yml` file  

```bash
conda env create -f environment.yml
```

Then activate it using the following command
```bash
conda activate env
```

Once is activated, you must run `nn_main.py` file passing as argument 20 values between **1-4** separated by a comma. Then
a prediction value **(0/1)** will be received.
```bash
python nn_main.py 1,2,3,4,1,1,4,3,1,2,1,2,3,4,2,1,3,4,1,2
1
```





## Folder Structure
```
📦hssad_NN
 ┣ 📂nn                           #Neural Network package 
 ┃ ┣ 📂activation                 #Activations functions logic 
 ┃ ┃ ┣ 📜__init__.py              #Each init file indicates python that current directory must be treated as package
 ┃ ┃ ┗ 📜activation_function.py
 ┃ ┣ 📂layer                      #Layers logic 
 ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┗ 📜layer.py
 ┃ ┣ 📂loss                       #Functions to calculate loss
 ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┗ 📜loss_function.py
 ┃ ┣ 📂optimizer                  #Optimizer to reduce the loss and adjust the weights
 ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┗ 📜optimizer.py
 ┃ ┣ 📂resources
 ┃ ┃ ┗ 📜dataset.csv
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜__main__.py                #Indicates to python that must execute this file when highest level
                                  package is executed from console via python -m nn
                                  
 ┃ ┗ 📜app.py                     #Is the entry point to the neural network implementation.The model and
                                   the necessary methods for saving,loading,training and predictionare defined here.
 ┣ 📜.gitignore                   # Ignores the specified files in the version control
 ┣ 📜environment.yml              # Saves the dependencies used in the project if is necessary 
                                  to replicate a new virtual environment
 ┣ 📜model.pickle                 # File where the model is saved after training phase, with all optimized weights.
 ┗ 📜nn_main.py                   # File that is executed by the server, receiving the user answers and returning a prediction value.
```




