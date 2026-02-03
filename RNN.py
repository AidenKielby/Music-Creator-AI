import math
import random
import pickle
import os
import numpy as np
import re

MAX_STATE_VALUE = 5.0
GRAD_CLIP_VALUE = 1.0


class NeuralNetwork:
    def __init__(self, inputNeurons: int, neuronsInHiddenLayer: int, outputNeurons: int, activationFunction: str):
        # this part should be self explanitory
        self.inputNeurons = inputNeurons
        self.neuronsPerHiddenLayer = neuronsInHiddenLayer

        self.hiddenWeights = self.initWeights(inputNeurons, neuronsInHiddenLayer)
        self.memoryWeights = self.initWeights(neuronsInHiddenLayer, neuronsInHiddenLayer)
        self.note_classes = max(1, outputNeurons - 1)
        self.noteWeights = self.initWeights(neuronsInHiddenLayer, self.note_classes)
        self.lengthWeights = self.initWeights(neuronsInHiddenLayer, 1).reshape(self.neuronsPerHiddenLayer)

        self.noteBiases = self.initBiases(self.note_classes)
        self.lengthBias = float(self.initBiases(1)[0])
        self.hiddenBiases = self.initBiases(neuronsInHiddenLayer)

        self.preActivations = []  # raw hidden values
        self.hiddenStates = []  # activated hidden values
        self.lengthOutputs = []
        self.noteErrors = []
        self.lengthErrors = []

        self.inputs = []
        self.uses = 0
        self.activationFunction = activationFunction

        # backprop stuff
        self.outputLayerErrors = []
        self.hiddenGradients = []
        self.inputGradients = []

    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def leakyReLU(self, x):
        return np.where(x > 0, x, x * 0.1)

    def leakyReLU_derivative(self, x):
        return np.where(x > 0, 1.0, 0.1)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1.0 - np.tanh(x) ** 2

    def softmax(self, x):
        shifted = x - np.max(x)
        exp = np.exp(shifted)
        denom = np.sum(exp) + 1e-12
        return exp / denom

    def useActivationFunction(self, x: float):
        if self.activationFunction == "Leaky_ReLU":
            return self.leakyReLU(x)
        if self.activationFunction == "Sigmoid":
            return self.sigmoid(x)
        if self.activationFunction == "Tanh":
            return self.tanh(x)
        
    def useActivationFunctionDeriviative(self, x: float):
        if self.activationFunction == "Leaky_ReLU":
            return self.leakyReLU_derivative(x)
        if self.activationFunction == "Sigmoid":
            sig = self.sigmoid(x)
            return sig * (1 - sig)
        if self.activationFunction == "Tanh":
            return self.tanh_derivative(x)

    def resetUses(self):
        self.uses = 0
        self.preActivations = []
        self.hiddenStates = []
        self.lengthOutputs = []
        self.noteErrors = []
        self.lengthErrors = []

    def initWeights(self, fan_in, fan_out):
        return np.random.uniform(-0.1, 0.1, (fan_in, fan_out))

    def initBiases(self, neurons: int):
        return np.random.uniform(-0.1, 0.1, neurons)
    
    def forwardPass(self, inputs: list[float]):
        inputs_arr = np.asarray(inputs, dtype=float).reshape(self.inputNeurons)
        self.inputs.append(inputs_arr)
        if self.uses == 0:
            lastActivtion = np.zeros(self.neuronsPerHiddenLayer)
        else:
            lastActivtion = np.asarray(self.preActivations[-1])
        
        raw_hidden = inputs_arr @ self.hiddenWeights + lastActivtion @ self.memoryWeights + self.hiddenBiases
        raw_hidden = np.clip(raw_hidden, -MAX_STATE_VALUE, MAX_STATE_VALUE)
        hiddenActivations = self.useActivationFunction(raw_hidden)
        
        self.preActivations.append(raw_hidden)
        self.hiddenStates.append(hiddenActivations)
        self.uses += 1

        note_logits = np.asarray(hiddenActivations) @ np.asarray(self.noteWeights) + np.asarray(self.noteBiases)
        note_probs = self.softmax(note_logits)

        length_raw = float(np.asarray(hiddenActivations) @ self.lengthWeights + self.lengthBias)
        length_value = self.sigmoid(length_raw)
        self.lengthOutputs.append(length_value)

        neuronOutput = np.concatenate([note_probs, [length_value]])

        return neuronOutput
    
    def backPropagate(self, expectedOutputs: list[list[float]], outputs: list[list[float]], learningRate: float):
        self.noteErrors = []
        self.lengthErrors = []
        self.outputLayerErrors = []

        for out_vec, target_vec in zip(outputs, expectedOutputs):
            note_probs = np.asarray(out_vec[:-1])
            target_notes = np.asarray(target_vec[:-1])
            length_pred = float(out_vec[-1])
            length_target = float(target_vec[-1])

            note_error = np.clip(note_probs - target_notes, -GRAD_CLIP_VALUE, GRAD_CLIP_VALUE)
            length_error = float(np.clip(length_pred - length_target, -GRAD_CLIP_VALUE, GRAD_CLIP_VALUE))

            self.noteErrors.append(note_error)
            self.lengthErrors.append(length_error)
            self.outputLayerErrors.append(np.concatenate([note_error, [length_error]]))

        noteWeightGradients = np.zeros_like(self.noteWeights)
        noteBiasGradients = np.zeros_like(self.noteBiases)
        lengthWeightGradients = np.zeros_like(self.lengthWeights)
        lengthBiasGradient = 0.0
        lengthDeltas: list[float] = []

        for hidden_act, note_err, length_err, length_pred in zip(self.hiddenStates, self.noteErrors, self.lengthErrors, self.lengthOutputs):
            noteWeightGradients += np.outer(hidden_act, note_err)
            noteBiasGradients += note_err

            sigmoid_prime = max(length_pred * (1 - length_pred), 1e-6)
            length_delta = length_err * sigmoid_prime
            lengthWeightGradients += hidden_act * length_delta
            lengthBiasGradient += length_delta
            lengthDeltas.append(length_delta)

        self.noteWeights -= noteWeightGradients * learningRate
        self.noteBiases -= noteBiasGradients * learningRate
        self.lengthWeights -= lengthWeightGradients * learningRate
        self.lengthBias -= lengthBiasGradient * learningRate

        futureError = np.zeros(self.neuronsPerHiddenLayer)
        recurrentWeightGrads = []
        inputWeightGrads = []
        for t in range(len(self.preActivations)-1, -1, -1):
            note_back = self.noteErrors[t] @ self.noteWeights.T
            length_back = lengthDeltas[t] * self.lengthWeights
            localError = note_back + length_back
            bothErrors = localError + self.memoryWeights.T @ futureError
            delta_hidden = bothErrors * self.useActivationFunctionDeriviative(self.preActivations[t])
            delta_hidden = np.clip(delta_hidden, -GRAD_CLIP_VALUE, GRAD_CLIP_VALUE)
            prev_hidden = np.zeros(self.neuronsPerHiddenLayer) if t == 0 else self.hiddenStates[t-1]
            recurrentWeightGrads.append(np.outer(prev_hidden, delta_hidden))
            inputWeightGrads.append(np.outer(self.inputs[t], delta_hidden))
            futureError = delta_hidden

        input_grad_sum = np.sum(inputWeightGrads, axis=0)
        self.memoryWeights -= np.sum(recurrentWeightGrads, axis=0) * learningRate
        self.hiddenWeights -= input_grad_sum * learningRate
        

def _demo_backpropagation():
    """Train a tiny network to flip a binary sequence as a sanity check."""
    random.seed(42)
    np.random.seed(42)

    rnn = NeuralNetwork(inputNeurons=1, neuronsInHiddenLayer=2, outputNeurons=2, activationFunction="Tanh")

    inputs = [[0.0], [0.0], [1.0], [1.0]]
    targets = [
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
    ]
    learning_rate = 0.3
    epochs = 200

    for epoch in range(epochs):
        rnn.resetUses()
        rnn.inputs = []
        rnn.outputLayerErrors = []

        outputs = [rnn.forwardPass(sample) for sample in inputs]

        rnn.backPropagate(targets, outputs, learning_rate)

        if epoch % 50 == 0 or epoch == epochs - 1:
            loss = float(np.mean([(out[-1] - tgt[-1]) ** 2 for out, tgt in zip(outputs, targets)]))
            print(f"Epoch {epoch:03d} | Length MSE: {loss:.6f}")

    rnn.resetUses()
    rnn.inputs = []
    final_outputs = [rnn.forwardPass(sample) for sample in inputs]
    print("Final sequence:")
    for original, predicted in zip(inputs, final_outputs):
        print(f"input {original[0]:.0f} -> note_prob={float(predicted[0]):.3f}, length={float(predicted[-1]):.3f}")


if __name__ == "__main__":
    _demo_backpropagation()

