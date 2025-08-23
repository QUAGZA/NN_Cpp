#pragma once
#include "NeuralNetwork.hpp"
#include <vector>

class Backpropagation {
public:
    static void train(
        NeuralNetwork& net,
        const std::vector<float>& input,
        const std::vector<float>& target,
        float learning_rate
    );

    static float computeLoss(
        const std::vector<float>& output,
        const std::vector<float>& target
    );
};
