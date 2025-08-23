#include <iostream>
#include <vector>
#include <SFML/Graphics.hpp>
#include "headers/NeuralNetwork.hpp"
#include "headers/Backpropagation.hpp"
#include "headers/NeuralNetVisualizer.hpp"

int main() {
    // XOR input and target data
    std::vector<std::vector<float>> inputs = {
        {0.f, 0.f},
        {0.f, 1.f},
        {1.f, 0.f},
        {1.f, 1.f}
    };
    std::vector<std::vector<float>> targets = {
        {1.f, 0.f}, // 0 XOR 0 = 0 (class 0)
        {0.f, 1.f}, // 0 XOR 1 = 1 (class 1)
        {0.f, 1.f}, // 1 XOR 0 = 1 (class 1)
        {1.f, 0.f}  // 1 XOR 1 = 0 (class 0)
    };

    sf::RenderWindow window(sf::VideoMode({800, 600}), "Neural Network Visualization");
    NeuralNetwork net({2, 4, 2});
    NeuralNetVisualizer visualizer(window, net);
    float learning_rate = 0.1f;
    int epochs = 10000;

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.f;
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto output = net.forward(inputs[i]);
            Backpropagation::train(net, inputs[i], targets[i], learning_rate);
            total_loss += Backpropagation::computeLoss(output, targets[i]);
        }
        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << " Loss: " << total_loss / inputs.size() << std::endl;
        }
    }

    // Test the trained network
    std::cout << "\nFinal outputs after training:\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = net.forward(inputs[i]);
        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] << "] "
                  << "Output: [" << output[0] << ", " << output[1] << "]" << std::endl;
    }

    while (window.isOpen()) {
        while (auto ev = window.pollEvent()) {
            if (ev->is<sf::Event::Closed>()) {
                window.close();
            }
        }
        window.clear(sf::Color::Black);
        visualizer.draw();
        window.display();
    }
    return 0;
}
