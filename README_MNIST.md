# MNIST Digit Recognition with C++ Neural Network

This project demonstrates how to train a neural network on the MNIST handwritten digit dataset using our C++ neural network implementation with SFML visualization.

## Prerequisites

- C++ compiler with C++17 support
- SFML 3 library installed
- The MNIST dataset files (see below)
- A TTF font file named `arial.ttf` in the project directory

## Getting the MNIST Dataset

Download the following files from [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/):

1. `train-images-idx3-ubyte.gz` - Training set images
2. `train-labels-idx1-ubyte.gz` - Training set labels
3. `t10k-images-idx3-ubyte.gz` - Test set images
4. `t10k-labels-idx1-ubyte.gz` - Test set labels

Extract these files (using gunzip or similar) and place them in the `data` directory with the following names:
- `data/train-images.idx3-ubyte`
- `data/train-labels.idx1-ubyte`
- `data/t10k-images.idx3-ubyte`
- `data/t10k-labels.idx1-ubyte`

## Compilation

You can compile the MNIST trainer using the provided script:

```bash
./compile_mnist.sh
```

Or manually:

```bash
g++ -std=c++17 train_mnist.cpp src/NeuralNetwork.cpp src/Backpropagation.cpp src/NeuralNetVisualizer.cpp -I/usr/include -L/usr/lib -lsfml-graphics -lsfml-window -lsfml-system -o train_mnist
```

## Running the Program

Execute the compiled program:

```bash
./train_mnist
```

## Network Architecture

The default neural network architecture for MNIST is:
- Input layer: 784 neurons (28×28 pixels)
- First hidden layer: 128 neurons
- Second hidden layer: 64 neurons
- Output layer: 10 neurons (one per digit 0-9)

## Training Parameters

The default training parameters are:
- Learning rate: 0.01
- Epochs: 5
- Batch size: 100
- Training samples: 10,000 (reduced for faster training, edit code to use all 60,000)

## Visualization

The visualization window shows:
- Left side: The current digit being processed and its predicted vs actual label
- Right side: Neural network visualization with neuron activations and connection weights

## Interactive Testing

After training completes, you can browse through the test set using:
- Left arrow key: Previous sample
- Right arrow key: Next sample

The window displays:
- The current digit image
- Predicted and actual labels
- Confidence values for each digit (0-9)
- The neural network state

## Expected Results

With the default settings, you should achieve approximately 90-95% accuracy on the test set. To improve accuracy:
1. Train for more epochs
2. Use more training samples
3. Adjust the learning rate
4. Add more neurons or layers

## Troubleshooting

If you encounter issues:
1. Ensure all MNIST files are downloaded and placed in the correct location
2. Make sure `arial.ttf` (or another TTF font) is present in the project directory
3. Check that SFML 3 is properly installed and linked

If the visualization is too slow, you can modify the display_interval to update the visualization less frequently.