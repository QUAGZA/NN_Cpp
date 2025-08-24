#!/bin/bash
g++ -std=c++17 train_mnist.cpp src/NeuralNetwork.cpp src/Backpropagation.cpp src/NeuralNetVisualizer.cpp -I/usr/include -L/usr/lib -lsfml-graphics -lsfml-window -lsfml-system -o train_mnist
echo "Compilation complete. Run with ./train_mnist"
echo "Make sure to download the MNIST dataset files into the data directory:"
echo "- data/train-images.idx3-ubyte"
echo "- data/train-labels.idx1-ubyte"
echo "- data/t10k-images.idx3-ubyte"
echo "- data/t10k-labels.idx1-ubyte"
echo ""
echo "You can download them from: http://yann.lecun.com/exdb/mnist/"
