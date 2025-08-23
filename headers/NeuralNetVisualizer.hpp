#ifndef NEURAL_NET_VISUALIZER_HPP
#define NEURAL_NET_VISUALIZER_HPP

#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics.hpp>
#include "NeuralNetwork.hpp"
#include <vector>

class NeuralNetVisualizer {
public:
    NeuralNetVisualizer(
        sf::RenderWindow& window,
        const NeuralNetwork& net,
        float neuronRadius = 20.f,
        float spacingX = 150.f,
        float spacingY = 80.f
    );

    void draw();

private:
    sf::RenderWindow& window;
    std::vector<int> layers;
    const NeuralNetwork& net;
    float neuronRadius;
    float spacingX;
    float spacingY;
    sf::Font font;

    std::vector<std::vector<sf::Vector2f>> neuronPositions;

    void calculateNeuronPositions();
    void drawConnections();
    void drawNeurons();
};

#endif
