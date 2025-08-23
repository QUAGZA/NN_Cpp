#include "../headers/Backpropagation.hpp"
#include <vector>
#include <cmath>

void Backpropagation::train(
    NeuralNetwork& net,
    const std::vector<float>& input,
    const std::vector<float>& target,
    float learning_rate
) {
    // Forward pass
    net.forward(input);

    const auto& layers = net.getLayers();
    auto& activations = net.getActivationsMutable();
    auto& weights = net.getWeightsMutable();
    auto& biases = net.getBiasesMutable();

    // Calculate output layer error (deltas)
    std::vector<std::vector<float>> deltas(layers.size());
    deltas.back().resize(layers.back());
    for (int j = 0; j < layers.back(); ++j) {
        float a = activations.back()[j];
        deltas.back()[j] = a - target[j];
    }

    // Backpropagate errors
    for (int l = layers.size() - 2; l > 0; --l) {
        deltas[l].resize(layers[l]);
        for (int i = 0; i < layers[l]; ++i) {
            float sum = 0.f;
            for (int j = 0; j < layers[l+1]; ++j) {
                sum += weights[l][i][j] * deltas[l+1][j];
            }
            float a = activations[l][i];
            deltas[l][i] = sum * a * (1 - a);
        }
    }

    // Update weights and biases
    for (int l = 0; l < layers.size() - 1; ++l) {
        for (int i = 0; i < layers[l]; ++i) {
            for (int j = 0; j < layers[l+1]; ++j) {
                weights[l][i][j] -= learning_rate * activations[l][i] * deltas[l+1][j];
            }
        }
        for (int j = 0; j < layers[l+1]; ++j) {
            biases[l][j] -= learning_rate * deltas[l+1][j];
        }
    }
}

float Backpropagation::computeLoss( // cross-entropy loss
    const std::vector<float>& output,
    const std::vector<float>& target
) {
    float loss = 0.f;
    for (size_t i = 0; i < output.size(); ++i) {
        loss -= target[i] * std::log(output[i] + 1e-8f); // Add epsilon to avoid log(0)
    }
    return loss / output.size();
}
