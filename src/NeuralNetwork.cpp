#include "../headers/NeuralNetwork.hpp"
#include <cmath>
#include <algorithm>

NeuralNetwork::NeuralNetwork(const std::vector<int>& layers)
    : layers(layers)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    // Initialize activations
    activations.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i)
        activations[i].resize(layers[i], 0.f);

    // Initialize weights and biases
    weights.resize(layers.size() - 1);
    biases.resize(layers.size() - 1);
    for (size_t l = 0; l < layers.size() - 1; ++l) {
        weights[l].resize(layers[l]);
        for (int i = 0; i < layers[l]; ++i) {
            weights[l][i].resize(layers[l+1]);
            for (int j = 0; j < layers[l+1]; ++j)
                weights[l][i][j] = dist(gen);
        }
        biases[l].resize(layers[l+1]);
        for (int j = 0; j < layers[l+1]; ++j)
            biases[l][j] = dist(gen);
    }
}

float NeuralNetwork::sigmoid(float x) const {
    return 1.f / (1.f + std::exp(-x));
}

std::vector<float> NeuralNetwork::softmax(const std::vector<float>& x) const {
    std::vector<float> result(x.size());
    float max_elem = *std::max_element(x.begin(), x.end());
    float sum = 0.f;
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::exp(x[i] - max_elem); // for numerical stability
        sum += result[i];
    }
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] /= sum;
    }
    return result;
}


std::vector<float> NeuralNetwork::forward(const std::vector<float>& input) {
    activations[0] = input;
    for (size_t l = 1; l < layers.size(); ++l) {
        for (int j = 0; j < layers[l]; ++j) {
            float sum = biases[l-1][j];
            for (int i = 0; i < layers[l-1]; ++i)
                sum += activations[l-1][i] * weights[l-1][i][j];
            if (l == layers.size() - 1) { // output layer will have softmax activation
                activations[l][j] = sum;
            } else {
                activations[l][j] = sigmoid(sum);
            }
        }
    }
    activations.back() = softmax(activations.back());
    return activations.back();
}
