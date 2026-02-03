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
        ]

        self.inputs = []
        self.uses = 0
        self.activationFunction = activationFunction

        # backprop stuff
        self.outputLayerErrors = []
        self.hiddenGradients = []
        self.inputGradients = []

    
    def sigmoid(self, x):
        return 1 / (1 + (math.pow(math.e, -x)))

    def leakyReLU(self, x):
        return x if x > 0 else x * 0.1

    def leakyReLU_derivative(self, x):
        return 1 if x > 0 else 0.1

    def useActivationFunction(self, x: float):
        if self.activationFunction == "Leaky_ReLU":
            return self.leakyReLU(x)
        if self.activationFunction == "Sigmoid":
            return self.sigmoid(x)
        
    def useActivationFunctionDeriviative(self, x: float):
        if self.activationFunction == "Leaky_ReLU":
            return self.leakyReLU_derivative(x)
        if self.activationFunction == "Sigmoid":
            return x * (1 - x) 

    def resetUses(self):
        self.uses = 0
        self.preActivations = [
        ]

    def initWeights(self, fan_in, fan_out):
        return np.random.uniform(-0.1, 0.1, (fan_in, fan_out))

    def initBiases(self, neurons: int):
        return np.random.uniform(-0.1, 0.1, neurons)
    
    def forwardPass(self, inputs: list[float]):
        self.inputs.append(inputs)
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
    
    def backPropagate(self, expectedOutput: list[float], lastOutput: list[float], learningRate: float):
        outputLayerError = np.asarray(lastOutput) - np.asarray(expectedOutput)
        self.outputLayerErrors.append(outputLayerError)

        PreActs = np.vstack(self.preActivations)
        outputErrors = np.vstack(self.outputLayerErrors)
        hiddenToOutputWeightGradients = np.transpose(outputErrors) @ PreActs

        self.weights = np.asmatrix(self.weights) - (np.asmatrix(hiddenToOutputWeightGradients) * learningRate)
        self.biases -= np.sum(outputErrors, axis=0) * learningRate

        futureError = 0
        recurrentWeightGrads = []
        inputWeightGrads = []
        for t in range(len(self.preActivations)-1, -1, -1):
            localError = np.transpose(np.asmatrix(self.weights)) @ self.outputLayerErrors[t]
            bothErrors = localError + self.memoryWeights.T @ futureError
            delta_hidden = bothErrors * self.useActivationFunctionDeriviative(self.preActivations[t])
            prev_hidden = np.zeros(self.neuronsPerHiddenLayer) if t == 0 else self.preActivations[t-1]
            recurrentWeightGrads.append(np.outer(delta_hidden, prev_hidden))
            inputWeightGrads.append(np.outer(delta_hidden, self.inputs[t]))
            futureError = delta_hidden

        self.memoryWeights = self.memoryWeights - (np.sum(recurrentWeightGrads, axis=0) * learningRate)
        self.hiddenWeights = self.hiddenWeights - (np.sum(inputWeightGrads, axis=0) * learningRate)
        
