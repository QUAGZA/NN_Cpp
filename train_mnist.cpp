#include <SFML/Window/Event.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <algorithm>
#include <SFML/Graphics.hpp>
#include "headers/NeuralNetwork.hpp"
#include "headers/Backpropagation.hpp"
#include "headers/NeuralNetVisualizer.hpp"

// MNIST dataset paths - adjust these to your actual file locations
const std::string TRAIN_IMAGES = "data/train-images.idx3-ubyte";
const std::string TRAIN_LABELS = "data/train-labels.idx1-ubyte";
const std::string TEST_IMAGES = "data/t10k-images.idx3-ubyte";
const std::string TEST_LABELS = "data/t10k-labels.idx1-ubyte";

// MNIST constants
const int IMAGE_MAGIC = 0x803;
const int LABEL_MAGIC = 0x801;
const int IMAGE_WIDTH = 28;
const int IMAGE_HEIGHT = 28;
const int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;
const int NUM_CLASSES = 10;

// MNIST dataset loading functions
uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xFF000000) |
           ((val << 8) & 0x00FF0000) |
           ((val >> 8) & 0x0000FF00) |
           ((val >> 24) & 0x000000FF);
}

bool load_mnist_images(const std::string& path, std::vector<std::vector<float>>& images) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << path << std::endl;
        return false;
    }

    uint32_t magic, num_images, rows, cols;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    magic = swap_endian(magic);
    num_images = swap_endian(num_images);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    if (magic != IMAGE_MAGIC) {
        std::cerr << "Invalid magic number in image file: " << magic << std::endl;
        return false;
    }

    std::cout << "Loading " << num_images << " images of size " << rows << "x" << cols << "..." << std::endl;

    images.resize(num_images);
    for (uint32_t i = 0; i < num_images; ++i) {
        images[i].resize(rows * cols);
        for (uint32_t j = 0; j < rows * cols; ++j) {
            uint8_t pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            // Normalize pixel value to [0, 1]
            images[i][j] = pixel / 255.0f;
        }
    }

    return true;
}

bool load_mnist_labels(const std::string& path, std::vector<std::vector<float>>& labels) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << path << std::endl;
        return false;
    }

    uint32_t magic, num_labels;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));

    magic = swap_endian(magic);
    num_labels = swap_endian(num_labels);

    if (magic != LABEL_MAGIC) {
        std::cerr << "Invalid magic number in label file: " << magic << std::endl;
        return false;
    }

    std::cout << "Loading " << num_labels << " labels..." << std::endl;

    labels.resize(num_labels);
    for (uint32_t i = 0; i < num_labels; ++i) {
        labels[i].resize(NUM_CLASSES, 0.0f);
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels[i][label] = 1.0f;  // One-hot encoding
    }

    return true;
}

// Helper function to shuffle data
void shuffle_data(std::vector<std::vector<float>>& images, std::vector<std::vector<float>>& labels) {
    std::random_device rd;
    std::mt19937 g(rd());

    // Create index vector
    std::vector<size_t> indices(images.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    // Shuffle indices
    std::shuffle(indices.begin(), indices.end(), g);

    // Create temporary vectors to hold shuffled data
    std::vector<std::vector<float>> shuffled_images(images.size());
    std::vector<std::vector<float>> shuffled_labels(labels.size());

    // Fill shuffled vectors
    for (size_t i = 0; i < indices.size(); ++i) {
        shuffled_images[i] = images[indices[i]];
        shuffled_labels[i] = labels[indices[i]];
    }

    // Copy back
    images = shuffled_images;
    labels = shuffled_labels;
}

// Helper function to compute accuracy
float compute_accuracy(NeuralNetwork& net, const std::vector<std::vector<float>>& images,
                      const std::vector<std::vector<float>>& labels) {
    int correct = 0;
    int total = std::min(images.size(), labels.size());

    for (int i = 0; i < total; ++i) {
        auto output = net.forward(images[i]);

        // Find predicted class (max output)
        int predicted_class = 0;
        float max_val = output[0];
        for (int j = 1; j < output.size(); ++j) {
            if (output[j] > max_val) {
                max_val = output[j];
                predicted_class = j;
            }
        }

        // Find true class (max label)
        int true_class = 0;
        max_val = labels[i][0];
        for (int j = 1; j < labels[i].size(); ++j) {
            if (labels[i][j] > max_val) {
                max_val = labels[i][j];
                true_class = j;
            }
        }

        if (predicted_class == true_class) {
            correct++;
        }
    }

    return static_cast<float>(correct) / total;
}

// Visualize a digit
void visualize_digit(const std::vector<float>& digit, sf::RenderWindow& window) {
    sf::Image image;
    image.create(IMAGE_WIDTH, IMAGE_HEIGHT);

    for (int y = 0; y < IMAGE_HEIGHT; ++y) {
        for (int x = 0; x < IMAGE_WIDTH; ++x) {
            uint8_t pixel_value = static_cast<uint8_t>(digit[y * IMAGE_WIDTH + x] * 255);
            image.setPixel(x, y, sf::Color(pixel_value, pixel_value, pixel_value));
        }
    }

    sf::Texture texture;
    texture.loadFromImage(image);
    sf::Sprite sprite(texture);

    float scale = 10.0f;  // Scale up for better visibility
    sprite.setScale(scale, scale);
    sprite.setPosition(50, 50);

    window.clear(sf::Color(50, 50, 50));
    window.draw(sprite);
}

int main() {
    // Network configuration
    std::vector<int> layers = {IMAGE_SIZE, 128, 64, NUM_CLASSES};
    NeuralNetwork net(layers);

    // SFML setup
    sf::RenderWindow window(sf::VideoMode({1200, 800}), "MNIST Neural Network Visualization");
    NeuralNetVisualizer visualizer(window, net);

    // Training parameters
    float learning_rate = 0.01f;
    int epochs = 5;
    int batch_size = 100;
    int display_interval = 100;

    // Load MNIST data
    std::vector<std::vector<float>> train_images;
    std::vector<std::vector<float>> train_labels;
    std::vector<std::vector<float>> test_images;
    std::vector<std::vector<float>> test_labels;

    if (!load_mnist_images(TRAIN_IMAGES, train_images) ||
        !load_mnist_labels(TRAIN_LABELS, train_labels) ||
        !load_mnist_images(TEST_IMAGES, test_images) ||
        !load_mnist_labels(TEST_LABELS, test_labels)) {
        std::cerr << "Failed to load MNIST data." << std::endl;
        std::cerr << "Please make sure the MNIST files are in the 'data' directory." << std::endl;
        // Download from: http://yann.lecun.com/exdb/mnist/
        return 1;
    }

    std::cout << "Data loaded successfully." << std::endl;
    std::cout << "Training samples: " << train_images.size() << std::endl;
    std::cout << "Testing samples: " << test_images.size() << std::endl;
    std::cout << "Network architecture: ";
    for (size_t i = 0; i < layers.size(); ++i) {
        std::cout << layers[i];
        if (i < layers.size() - 1) std::cout << "-";
    }
    std::cout << std::endl;

    // Limit training data for faster training (optional)
    const size_t max_training_samples = 10000;  // Use 10k samples
    if (train_images.size() > max_training_samples) {
        train_images.resize(max_training_samples);
        train_labels.resize(max_training_samples);
        std::cout << "Limited training to " << max_training_samples << " samples for faster training." << std::endl;
    }

    // Shuffle data
    shuffle_data(train_images, train_labels);

    // Training loop
    std::cout << "Starting training..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    int total_batches = train_images.size() / batch_size;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;

        for (size_t batch_start = 0; batch_start < train_images.size(); batch_start += batch_size) {
            size_t batch_end = std::min(batch_start + batch_size, train_images.size());
            float batch_loss = 0.0f;

            for (size_t i = batch_start; i < batch_end; ++i) {
                auto output = net.forward(train_images[i]);
                Backpropagation::train(net, train_images[i], train_labels[i], learning_rate);
                batch_loss += Backpropagation::computeLoss(output, train_labels[i]);

                // Process SFML events to keep the window responsive
                // while (auto ev = window.pollEvent()) {
                //     if (ev->is<sf::Event::Closed>()) {
                //         window.close();
                //         return 0;
                //     }
                // }
                sf::Event ev;
                while (window.pollEvent(ev)) {
                    if (ev.type == sf::Event::Closed) {
                        window.close();
                        return 0;
                    }
                    // ... handle other events
                }
            }

            epoch_loss += batch_loss;

            // Display progress and update visualization periodically
            int batch_num = batch_start / batch_size + 1;
            if (batch_num % display_interval == 0 || batch_num == total_batches) {
                float avg_batch_loss = batch_loss / (batch_end - batch_start);
                float progress = static_cast<float>(batch_num) / total_batches * 100;
                std::cout << "Epoch " << epoch + 1 << "/" << epochs
                          << " - Batch " << batch_num << "/" << total_batches
                          << " (" << progress << "%) - Loss: " << avg_batch_loss << std::endl;

                // Update visualization - show a sample digit and network state
                if (window.isOpen()) {
                    // Forward pass on a sample for visualization
                    size_t sample_idx = batch_start % train_images.size();
                    net.forward(train_images[sample_idx]);

                    window.clear(sf::Color::Black);

                    // Split the window: left for digit, right for network
                    sf::View leftView(sf::FloatRect(0, 0, 400, 800));
                    leftView.setViewport(sf::FloatRect(0, 0, 0.33f, 1));
                    window.setView(leftView);

                    // Draw the current digit being processed
                    visualize_digit(train_images[sample_idx], window);

                    // Draw the predicted label
                    sf::Font font;
                    if (!font.loadFromFile("arial.ttf")) {
                        std::cerr << "Failed to load font!" << std::endl;
                    } else {
                        sf::Text text;
                        text.setFont(font);
                        auto output = net.forward(train_images[sample_idx]);
                        int predicted = std::max_element(output.begin(), output.end()) - output.begin();
                        int actual = std::max_element(train_labels[sample_idx].begin(),
                                                    train_labels[sample_idx].end()) -
                                    train_labels[sample_idx].begin();
                        text.setString("Predicted: " + std::to_string(predicted) +
                                      "\nActual: " + std::to_string(actual));
                        text.setCharacterSize(24);
                        text.setFillColor(sf::Color::White);
                        text.setPosition(50, 350);
                        window.draw(text);
                    }

                    // Switch to right view for network visualization
                    sf::View rightView(sf::FloatRect(0, 0, 800, 800));
                    rightView.setViewport(sf::FloatRect(0.33f, 0, 0.67f, 1));
                    window.setView(rightView);

                    // Draw neural network
                    visualizer.draw();

                    window.display();
                }
            }
        }

        // Calculate metrics at the end of each epoch
        float avg_epoch_loss = epoch_loss / train_images.size();
        float train_accuracy = compute_accuracy(net, train_images, train_labels);
        float test_accuracy = compute_accuracy(net, test_images, test_labels);

        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << " - Loss: " << avg_epoch_loss
                  << " - Train Acc: " << train_accuracy * 100 << "%"
                  << " - Test Acc: " << test_accuracy * 100 << "%" << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Training completed in " << duration.count() << " seconds." << std::endl;

    // Final evaluation
    float final_accuracy = compute_accuracy(net, test_images, test_labels);
    std::cout << "Final test accuracy: " << final_accuracy * 100 << "%" << std::endl;

    // Interactive testing loop
    std::cout << "Starting interactive test visualization. Close the window to exit." << std::endl;
    size_t current_test_idx = 0;

    while (window.isOpen()) {
        sf::Event ev;
        while (window.isOpen()) {
            sf::Event ev;
            while (window.pollEvent(ev)) {
                if (ev.type == sf::Event::Closed) {
                    window.close();
                } else if (ev.type == sf::Event::KeyPressed) {
                    // ev.key is a struct, not a pointer
                    if (ev.key.code == sf::Keyboard::Right) {
                        // Next test sample
                        current_test_idx = (current_test_idx + 1) % test_images.size();
                    } else if (ev.key.code == sf::Keyboard::Left) {
                        // Previous test sample
                        current_test_idx = (current_test_idx + test_images.size() - 1) % test_images.size();
                    }
                }
            }
        }


        // Process current test sample
        auto output = net.forward(test_images[current_test_idx]);

        // Split view for visualization
        window.clear(sf::Color::Black);

        // Left side - digit
        sf::View leftView(sf::FloatRect(0, 0, 400, 800));
        leftView.setViewport(sf::FloatRect(0, 0, 0.33f, 1));
        window.setView(leftView);
        visualize_digit(test_images[current_test_idx], window);

        // Draw prediction info
        sf::Font font;
        if (font.loadFromFile("arial.ttf")) {
            sf::Text text;
            text.setFont(font);
            int predicted = std::max_element(output.begin(), output.end()) - output.begin();
            int actual = std::max_element(test_labels[current_test_idx].begin(),
                                         test_labels[current_test_idx].end()) -
                         test_labels[current_test_idx].begin();

            text.setString("Predicted: " + std::to_string(predicted) +
                          "\nActual: " + std::to_string(actual) +
                          "\n\nSample: " + std::to_string(current_test_idx) +
                          "\n\nUse arrow keys\nto navigate");
            text.setCharacterSize(24);
            text.setFillColor(sf::Color::White);
            text.setPosition(50, 350);
            window.draw(text);

            // Also draw confidence values
            sf::Text confidenceText;
            confidenceText.setFont(font);
            std::string confStr = "Confidence:\n";
            for (int i = 0; i < 10; ++i) {
                confStr += std::to_string(i) + ": " + std::to_string(output[i]) + "\n";
            }
            confidenceText.setString(confStr);
            confidenceText.setCharacterSize(16);
            confidenceText.setFillColor(sf::Color::White);
            confidenceText.setPosition(50, 550);
            window.draw(confidenceText);
        }

        // Right side - neural network
        sf::View rightView(sf::FloatRect(0, 0, 800, 800));
        rightView.setViewport(sf::FloatRect(0.33f, 0, 0.67f, 1));
        window.setView(rightView);
        visualizer.draw();

        window.display();

        // Small delay to prevent high CPU usage
        sf::sleep(sf::milliseconds(50));
    }

    return 0;
}
