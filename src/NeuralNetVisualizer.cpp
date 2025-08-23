#include "../headers/NeuralNetVisualizer.hpp"
#include <SFML/Graphics/Rect.hpp>
#include <SFML/System/Vector2.hpp>
#include <SFML/Graphics.hpp>
#include <cstdint>
#include <iostream>

NeuralNetVisualizer::NeuralNetVisualizer(
    sf::RenderWindow& window,
    const NeuralNetwork& net,
    float neuronRadius,
    float spacingX,
    float spacingY
) : window(window), net(net), neuronRadius(neuronRadius),
    spacingX(spacingX), spacingY(spacingY), layers(net.getLayers())
{
    if (!font.openFromFile("arial.ttf")) {
        // Handle error: you can print a message or exit
        std::cout << "Failed to load font!" << std::endl;
    }
    calculateNeuronPositions();
}

void NeuralNetVisualizer::calculateNeuronPositions() {
    neuronPositions.clear();

    float startX = 100.f;
    float startY = 100.f;

    for (size_t i = 0; i < layers.size(); i++) {
        std::vector<sf::Vector2f> layerPositions;
        float offsetY = (window.getSize().y - (layers[i] * spacingY)) / 2.f;

        for (int j = 0; j < layers[i]; j++) {
            layerPositions.push_back(sf::Vector2f(startX + i * spacingX, offsetY + j * spacingY));
        }

        neuronPositions.push_back(layerPositions);
    }
}

void NeuralNetVisualizer::drawConnections() {
    const auto& weights = net.getWeights();
    sf::VertexArray lines(sf::PrimitiveType::Lines);

    for (size_t l = 0; l < neuronPositions.size() - 1; ++l) {
        for (size_t i = 0; i < neuronPositions[l].size(); ++i) {
            for (size_t j = 0; j < neuronPositions[l+1].size(); ++j) {
                float w = weights[l][i][j];
                // Map weight to color: blue for positive, red for negative
                sf::Color color = (w >= 0)
                    ? sf::Color(100, 100, 255, static_cast<uint8_t>(std::min(255.f, w * 255)))
                    : sf::Color(255, 100, 100, static_cast<uint8_t>(std::min(255.f, -w * 255)));
                lines.append(sf::Vertex{neuronPositions[l][i], color});
                lines.append(sf::Vertex{neuronPositions[l+1][j], color});
            }
        }
    }
    window.draw(lines);
}

// void NeuralNetVisualizer::drawConnections() {
//     sf::VertexArray lines(sf::PrimitiveType::Lines);

//     for (size_t i = 0; i < neuronPositions.size() - 1; i++) {
//         for (auto& n1 : neuronPositions[i]) {
//             for (auto& n2 : neuronPositions[i + 1]) {
//                 lines.append(sf::Vertex{n1, sf::Color::White});
//                 lines.append(sf::Vertex{n2, sf::Color::White});
//             }
//         }
//     }

//     window.draw(lines);
// }

void NeuralNetVisualizer::drawNeurons() {
    const auto& activations = net.getActivations();
    for (size_t l = 0; l < neuronPositions.size(); ++l) {
        for (size_t n = 0; n < neuronPositions[l].size(); ++n) {
            sf::CircleShape neuron(neuronRadius);
            neuron.setPosition(sf::Vector2f(neuronPositions[l][n].x - neuronRadius, neuronPositions[l][n].y - neuronRadius));
            float act = activations[l][n];
            // Map activation [0,1] to color intensity (e.g., blue)
            uint8_t intensity = static_cast<uint8_t>(act * 255);
            neuron.setFillColor(sf::Color(100, 200, intensity));
            neuron.setOutlineColor(sf::Color::Black);
            neuron.setOutlineThickness(2.f);

            window.draw(neuron);

            sf::Text text(font);
            char buffer[8];
            snprintf(buffer, sizeof(buffer), "%.2f", act); // Format to 2 decimal places
            text.setString(buffer);
            text.setCharacterSize(14);
            text.setFillColor(sf::Color::Black);
            // Center the text on the neuron
            sf::FloatRect textRect = text.getLocalBounds();
            text.setOrigin(textRect.getCenter());
            text.setPosition(sf::Vector2f(neuronPositions[l][n].x, neuronPositions[l][n].y));

            window.draw(text);
        }
    }
}

void NeuralNetVisualizer::draw() {
    drawConnections();
    drawNeurons();
}
