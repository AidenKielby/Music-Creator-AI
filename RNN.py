import math
import random
import pickle
import os
import numpy as np
import re

class NeuralNetwork:
    def __init__(self, inputNeurons: int, neuronsInHiddenLayer: int, outputNeurons: int, activationFunction: str):
        # this part should be self explanitory
        self.inputNeurons = inputNeurons
        self.neuronsPerHiddenLayer = neuronsInHiddenLayer

        self.hiddenWeights = self.initWeights(inputNeurons, neuronsInHiddenLayer)
        self.memoryWeights = self.initWeights(neuronsInHiddenLayer, neuronsInHiddenLayer)
        self.weights = self.initWeights(neuronsInHiddenLayer, outputNeurons)

        self.biases = self.initBiases(outputNeurons)
        self.hiddenBiases = self.initBiases(neuronsInHiddenLayer)

        self.preActivations = [
            #One  of these for each use of RNN
            #        | | |
            #        v v v 
            []
        ]
        self.uses = 0
        self.activationFunction = activationFunction
    
    def sigmoid(self, x):
        return 1 / (1 + (math.pow(math.e, -x)))

    def leakyReLU(self, x):
        return x if x > 0 else x * 0.1

    def leakyReLU_derivative(self, x):
        return 1 if x > 0 else 0.1

    def useActivationFunction(self, x: int):
        if self.activationFunction == "Leaky_ReLU":
            return self.leakyReLU(x)
        if self.activationFunction == "Sigmoid":
            return self.sigmoid(x)

    def resetUses(self):
        self.uses = 0
        self.preActivations = [
            #One  of these for each use of RNN
            #        | | |
            #        v v v 
            [[] for x in range(self.hiddenLayers + 2)]
        ]

    def initWeights(self, fan_in, fan_out):
        return np.random.uniform(-0.1, 0.1, (fan_in, fan_out))

    def initBiases(self, neurons: int):
        return np.random.uniform(-0.1, 0.1, neurons)
    
    def forewardPass(self, inputs: list[float]):
        if self.uses == 0:
            lastActivtion = np.zeros(self.neuronsPerHiddenLayer)
        else:
            lastActivtion = np.asarray(self.preActivations[-1])
        
        hiddenActivations = self.useActivationFunction(np.asarray(self.hiddenWeights) @ np.asarray(inputs) + 
                                                       np.asarray(self.memoryWeights) @ np.asarray(lastActivtion) + 
                                                       np.asarray(self.hiddenBiases))
        
        self.preActivations.append(hiddenActivations)
        self.uses += 1

        neuronOutput = self.useActivationFunction(np.asarray(hiddenActivations) @ np.asarray(self.weights) + np.asarray(self.biases))

        return neuronOutput





    

