# Single-Layer Perceptron for Digit Classification

This repository contains MATLAB code for a Single-Layer Perceptron (SLP) used to classify handwritten digits (0-9). The implementation includes a gradient descent algorithm and generates a confusion matrix to evaluate the model's performance.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Performance Evaluation](#performance-evaluation)
- [License](#license)

## Introduction

The perceptron is a fundamental building block in neural networks. This project demonstrates the use of a single-layer perceptron (monocouche) for classifying digits from a given dataset. The classification is achieved using a gradient descent algorithm, and the model's accuracy is evaluated using a confusion matrix.

## Project Structure

- `perceptron.m`: Implements the perceptron learning algorithm with gradient descent.
- `digit.m`: Loads the dataset of handwritten digits and prepares it for training and evaluation.
- `monocouche.m`: Contains the main code that orchestrates the perceptron training and evaluation process using a single-layer architecture.

## Installation

To run the project locally, you need to have MATLAB installed. You can download MATLAB from the official website: [https://www.mathworks.com/products/matlab.html](https://www.mathworks.com/products/matlab.html).

### Clone the Repository
```bash
git clone https://github.com/your-username/perceptron-digit-classification.git
cd perceptron-digit-classification
```

## Usage

1. Ensure all necessary files (perceptron.m, digit.m, monocouche.m) are in the same directory.
2. Open MATLAB and navigate to the project directory.
3. Run monocouche.m to start the digit classification process.

The training process will involve the following steps:

- Load the dataset from digit.m.
- Initialize the perceptron model with parameters defined in perceptron.m.
- Train the model using gradient descent.
- Evaluate the model by generating a confusion matrix.

### Monocouche usage
The `monocouche.m` script starts up a menu in which you can choose whether you want to train the model or test it using validation data. After training, the weights of the model are saved in a separate file, and loaded upon testing. So make sure that you are testing the model with the right weights saved! If you are unsure, re-train the model and then test it.

## Example

To run the perceptron and classify digits, use:
```matlab
monocouche
```
## Data

This project uses two types of datasets located in specific folders:

- `Data/`: This folder contains the dataset of handwritten digits (0-9) that is used for training and testing the perceptron model. The data is organized in a format compatible with digit.m for loading into MATLAB.
- `DataSimulation/`: This folder contains a dataset for 2-dimensional inputs. These inputs are typically used for binary classification or to demonstrate the perceptron’s decision boundary. This data is useful for understanding how the perceptron operates with simpler, low-dimensional data.

Ensure that both folders and their respective datasets are in the working directory before running the MATLAB scripts.

## Files

- `perceptron.m`: The perceptron training code with gradient descent. It includes the following steps:
  - Weight initialization.
  - Update rule for the perceptron.
  - Training loop with the number of iterations.
- `digit.m`: Binary classification of two digits using a single perceptron.
- `monocouche.m`: Main execution script that runs the perceptron training and evaluates the model’s performance for as many digits as you want.

## Performance Evaluation

Once the perceptron has been trained, the model’s performance is evaluated by generating a confusion matrix. The confusion matrix provides insights into how well the model classifies each digit and helps to visualize the accuracy and errors made by the perceptron.

You can adjust the following parameters to improve the classification accuracy:

- Learning rate.
- Number of iterations (epochs).
- Initial weight values.

## Current Results

In the current implementation, with Nb = 500 gradient descent iterations, the model is able to classify all 10 digits (0-9) with a reasonable accuracy.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.
