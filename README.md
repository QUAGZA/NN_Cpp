# Neural Nets in C++ !!!!

Constructed a neural network from scratch in C++ and added a visualization function too.

## Overview

This project demonstrates a simple neural network framework implemented entirely in C++. It supports forward propagation, backward propagation (backpropagation), customizable activation functions, and includes a visualization module for inspecting network architecture and training progress (note: visualization may be buggy for larger networks).

A separate training script/file is provided to train the network on the MNIST digit recognition dataset.

## Features

- **Pure C++ Implementation:** No deep learning libraries required.
- **Forward & Backward Propagation:** Core neural network training logic.
- **Custom Activation Functions:** Easily switch between activation types.
- **Visualization:** Inspect network structure and weights (buggy for large networks).
- **MNIST Training:** Example setup for digit classification using MNIST.

## Getting Started

### Prerequisites

- **C++ Compiler:** GCC, Clang, or MSVC (C++11 or later)
- **CMake** (recommended) or direct compiler invocation
- **Git** (for cloning)
- **MNIST Dataset:** Download from [here](http://yann.lecun.com/exdb/mnist/) or use the provided script to fetch it.
- **(Optional) Shell:** Some scripts may use shell for automation (tested on Bash and Windows Git Bash).

### Cloning the Repository

```bash
git clone https://github.com/QUAGZA/NN_Cpp-.git
cd NN_Cpp-
```

### Build Instructions

#### Windows

1. Install [MinGW](http://www.mingw.org/) or use Visual Studio.
2. Open a terminal (CMD/PowerShell or Git Bash).
3. If using CMake:

    ```bash
    mkdir build
    cd build
    cmake ..
    cmake --build .
    ```

   Or, directly compile (example):

    ```bash
    g++ src/*.cpp -o nn_cpp.exe
    ```

#### macOS & Linux

1. Make sure `g++` or `clang++` is installed.
2. Open a terminal.
3. If using CMake:

    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

   Or, directly compile:

    ```bash
    g++ src/*.cpp -o nn_cpp
    ```

#### Notes

- Replace `src/*.cpp` with actual source file locations if necessary.
- Visualization might require additional dependencies (e.g., SFML, OpenGL, or similar) depending on implementation. See `src/visualization.cpp` for details.
- For training MNIST, ensure the dataset files are present or run the provided shell script to download.

### Training the Network on MNIST

1. Prepare the MNIST data (follow instructions in `train_mnist.sh` or `train_mnist.bat`).
2. Run the training script:

    ```bash
    ./train_mnist
    ```

   or (Windows):

    ```bash
    train_mnist.exe
    ```

3. Follow prompts or edit configuration as needed.

### Visualizing the Network

- After or during training, run the visualization module with the appropriate command:

    ```bash
    ./visualize_network
    ```

   (See documentation in `src/visualization.cpp` for options and limitations.)

## Bugs & Limitations

- Visualization may not render correctly for larger networks.
- MNIST training is basic; for advanced tasks, further optimization and regularization are recommended.

## Contributing

Bug reports and pull requests are welcome!

## License

This project is licensed under the MIT License.

## Acknowledgements

- MNIST dataset by Yann LeCun et al.
- Inspiration from classic neural network tutorials.
