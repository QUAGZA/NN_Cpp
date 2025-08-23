#pragma once
#include <vector>
#include <random>

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layers);

    // Forward pass: input -> output
    std::vector<float> forward(const std::vector<float>& input);

    // Accessors for visualization
    const std::vector<std::vector<float>>& getActivations() const { return activations; }
    const std::vector<std::vector<std::vector<float>>>& getWeights() const { return weights; }

    const std::vector<int>& getLayers() const { return layers; }
    std::vector<std::vector<float>>& getActivationsMutable() { return activations; }
    std::vector<std::vector<std::vector<float>>>& getWeightsMutable() { return weights; }
    std::vector<std::vector<float>>& getBiasesMutable() { return biases; }
    std::vector<float> softmax(const std::vector<float>& x) const;
private:
    std::vector<int> layers;
    std::vector<std::vector<float>> activations; // [layer][neuron]
    std::vector<std::vector<std::vector<float>>> weights; // [layer][from][to]
    std::vector<std::vector<float>> biases; // [layer][neuron]

    float sigmoid(float x) const;
    float randomWeight();
};
