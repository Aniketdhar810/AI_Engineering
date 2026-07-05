# Artificial Neural Networks (ANN) Reference Guide

This directory contains notebooks and reference materials detailing the fundamentals of Artificial Neural Networks, activation functions, backpropagation, and training optimization.

---

## 1. Deep Learning Applications
Deep learning is one of the hottest fields in data science, enabling breakthrough applications including:
* **Color Restoration**: Automatically converting grayscale images into realistic colored images.
* **Speech Enactment**: Synthesizing audio clips with corresponding lip movements in videos, extracting audio from one video and syncing it with lip movements from another.
* **Handwriting Generation**: Rewriting a text message in highly realistic cursive handwriting across a variety of styles.

---

## 2. Biological vs. Artificial Neurons

Deep learning algorithms are largely inspired by the biological structure and processing mechanisms of the human brain.

### The Biological Neuron
* **Soma**: The main body of the neuron where information processing occurs.
* **Dendrites**: An extensive network of branching arms sticking out of the soma. They receive electrical impulses from other adjoining neurons and carry them to the soma.
* **Axon**: A single long arm extending from the soma in the opposite direction of the dendrites, carrying processed electrical impulses away from the soma.
* **Synapses**: Whisker-like structures at the end of the axon that transfer outputs from the current neuron to the dendrites of thousands of neighboring neurons.
* **Brain Learning**: Occurs by repeatedly activating specific neural pathways over others, reinforcing those connections.

```
                  +-------------------+
  Dendrites ----> |  Soma (Nucleus)   | ----> Axon ----> Synapses
  (Inputs)        | (Processing/Sum)  |       (Output)   (To next neurons)
                  +-------------------+
```

### The Artificial Neuron
An artificial neuron models the behavior of its biological counterpart:
* **Input Layer**: The first layer that feeds raw data features into the neural network.
* **Hidden Layers**: Sets of nodes between the input and output layers that extract abstract features.
* **Output Layer**: The final set of nodes that output predictions.
* **Forward Propagation**: The process of passing data forward through layers of neurons (using weights and biases) from the input layer to the output layer to compute the network output.

---

## 3. Optimization & Training

### Gradient Descent & Learning Rates
Gradient descent is an iterative optimization algorithm used to find the minimum of a loss function.
* **Large Learning Rate**: Takes large steps, risking overshooting and missing the minimum point entirely.
* **Small Learning Rate**: Takes extremely small steps, causing the training process to take a long time to converge.

### The Neural Network Training Loop
Weights and biases are initialized to random values. The following four-step process is repeated in a loop:
1. **Forward Propagation**: Compute the network output for a given input.
2. **Error Calculation**: Calculate the difference (loss) between the ground truth and the predicted output.
3. **Backpropagation**: Pass the error backward through the network, calculating gradients and updating the weights and biases using gradient descent.
4. **Repeat**: Loop the previous steps until a maximum number of epochs is reached, or the error falls below a predefined threshold.

---

## 4. The Vanishing Gradient Problem

* **Cause**: Primarily caused by using activation functions like **Sigmoid** in deep hidden layers.
* **Mechanism**: During backpropagation, error gradients are multiplied layer-by-layer moving backward. If these gradients are repeatedly multiplied by factors less than $1$ (like the derivative of Sigmoid, which is $\le 0.25$), they shrink exponentially.
* **Effects**:
  * The error gradient with respect to early weights (e.g., $w_1$) becomes extremely small.
  * Neurons in the earlier layers learn much slower than neurons in the later layers.
  * Because the early layers (which extract basic features) are the slowest to train, overall training takes too long and final prediction accuracy is compromised.
* **Solution**: Avoid using sigmoid or similar functions as activation functions in hidden layers.

---

## 5. Activation Functions

Activation functions introduce non-linearity into the network, allowing it to learn complex, non-linear relationships.

* **Hyperbolic Tangent ($\tanh$)**:
  * A scaled version of the sigmoid function, but symmetric over the origin.
  * Outputs values between $-1$ and $+1$ (zero-centered, which aids faster training convergence).
* **Rectified Linear Unit (ReLU)**:
  * The most widely used activation function in modern neural networks.
  * **Formula**: $f(x) = \max(0, x)$
  * **Main Advantage**: Sparsity. It outputs $0$ for negative inputs, meaning it does not activate all neurons simultaneously, which reduces computation.
* **Softmax**:
  * Ideally used in the **output layer** of a classification model.
  * Converts raw scores (logits) into a probability distribution representing the likelihood of each class, ensuring the sum of all outputs equals $1$.
