#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Change to the data directory
cd data

# Download MNIST dataset
echo "Downloading MNIST dataset..."

# Training set images
if [ ! -f "train-images.idx3-ubyte" ]; then
    echo "Downloading training images..."
    wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    gunzip train-images-idx3-ubyte.gz
    mv train-images-idx3-ubyte train-images.idx3-ubyte
fi

# Training set labels
if [ ! -f "train-labels.idx1-ubyte" ]; then
    echo "Downloading training labels..."
    wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    gunzip train-labels-idx1-ubyte.gz
    mv train-labels-idx1-ubyte train-labels.idx1-ubyte
fi

# Test set images
if [ ! -f "t10k-images.idx3-ubyte" ]; then
    echo "Downloading test images..."
    wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    gunzip t10k-images-idx3-ubyte.gz
    mv t10k-images-idx3-ubyte t10k-images.idx3-ubyte
fi

# Test set labels
if [ ! -f "t10k-labels.idx1-ubyte" ]; then
    echo "Downloading test labels..."
    wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    gunzip t10k-labels-idx1-ubyte.gz
    mv t10k-labels-idx1-ubyte t10k-labels.idx1-ubyte
fi

echo "Download complete! All MNIST files are in the data directory."
echo "Files downloaded:"
ls -la

cd ..

# Check if arial.ttf exists, if not, provide instructions
if [ ! -f "arial.ttf" ]; then
    echo ""
    echo "Note: The font file 'arial.ttf' is not found in the current directory."
    echo "This file is needed for text rendering in the visualization."
    echo "Please copy an Arial TTF font file to this directory and name it 'arial.ttf'."
fi

echo ""
echo "Next steps:"
echo "1. Run ./compile_mnist.sh to compile the MNIST trainer"
echo "2. Run ./train_mnist to start training"
