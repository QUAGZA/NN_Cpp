#include <SFML/Graphics.hpp>
#include "headers/NeuralNetVisualizer.hpp"

int main() {
    sf::RenderWindow window(sf::VideoMode({800, 600}), "Neural Network Visualizer (SFML 3)");

    std::vector<int> layers = {3, 5, 2}; // Example: 3 input, 5 hidden, 2 output
    NeuralNetVisualizer visualizer(window, layers);

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
