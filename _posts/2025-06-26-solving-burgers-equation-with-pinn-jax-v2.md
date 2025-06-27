---
layout: post
title: "Solving the 1D Burgers' Equation with Physics-Informed Neural Networks (PINNs) in JAX (v2)"
date: 2025-06-27 10:00:00 -0700
categories: [Physics, Machine Learning, JAX, PINNs]
---

The Burgers' equation is a fundamental partial differential equation (PDE) that models the propagation of shock waves. It's a simplified model for fluid dynamics and is widely used as a test case for numerical methods. In this post, we'll explore how to solve the 1D Burgers' equation using Physics-Informed Neural Networks (PINNs) implemented in JAX.

## The 1D Burgers' Equation

The 1D Burgers' equation is given by:

$$ \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0 $$

where $u(t, x)$ is the velocity field, $t$ is time, $x$ is space, and $\nu$ is the kinematic viscosity coefficient.

We'll solve this equation with the following initial and boundary conditions:

*   **Initial Condition (IC):** $u(0, x) = -\sin(\pi x)$ for $x \in [-1, 1]$
*   **Boundary Conditions (BCs):** $u(t, -1) = u(t, 1) = 0$ for $t \in [0, 1]$

## Physics-Informed Neural Networks (PINNs)

PINNs are neural networks that are trained to satisfy both observed data and the underlying physical laws (expressed as PDEs). The core idea is to incorporate the PDE directly into the neural network's loss function.

In our case, the neural network will approximate the solution $u(t, x)$. We'll define a loss function that has two main components:

1.  **Data Loss (Mean Squared Error):** This term ensures that the neural network's predictions match the known initial and boundary conditions.
2.  **Physics Loss (Mean Squared Error):** This term enforces the Burgers' equation. We compute the derivatives of the neural network's output with respect to $t$ and $x$ using automatic differentiation (a powerful feature of JAX) and then ensure that these derivatives satisfy the Burgers' equation.

## Neural Network Architecture

Our neural network is a simple Multi-Layer Perceptron (MLP) with 5 hidden layers, each containing 20 neurons. The input to the network is a 2-dimensional vector $[t, x]$, and the output is the scalar value $u(t, x)$. We use the hyperbolic tangent (tanh) activation function for the hidden layers.

```python
class MLP(nn.Module):
    features: list[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.tanh(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x
```

## Loss Function

The total loss function for our PINN is:

$$ L = L_{data} + L_{physics} $$

where:

*   $L_{data} = \frac{1}{N_{data}} \sum_{i=1}^{N_{data}} (u_{pred}(t_i, x_i) - u_{true}(t_i, x_i))^2$
    This is the mean squared error between the network's predictions and the known values at the initial and boundary conditions.

*   $L_{physics} = \frac{1}{N_{collocation}} \sum_{j=1}^{N_{collocation}} (\frac{\partial u_{pred}}{\partial t} + u_{pred} \frac{\partial u_{pred}}{\partial x} - \nu \frac{\partial^2 u_{pred}}{\partial x^2})^2$
    This is the mean squared error of the Burgers' equation residual at a set of randomly sampled collocation points within the domain.

JAX's automatic differentiation (`jax.grad`) is used to compute the necessary derivatives ($\frac{\partial u}{\partial t}$, $\frac{\partial u}{\partial x}$, $\frac{\partial^2 u}{\partial x^2}$). The `jax.vmap` function is crucial for efficiently computing these derivatives over batches of input points.

## Training Process

The training process involves minimizing the total loss function using an Adam optimizer. We iteratively update the neural network's parameters based on the gradients of the loss.

1.  **Data Generation:**
    *   **Initial Condition Points ($N_{b}/2$):** Sample $x$ values at $t=0$ and use the initial condition $u(0, x) = -\sin(\pi x)$.
    *   **Boundary Condition Points ($N_{b}/2$ for each boundary):** Sample $t$ values at $x=-1$ and $x=1$, with $u(t, -1) = u(t, 1) = 0$.
    *   **Collocation Points ($N_{c}$):** Randomly sample $(t, x)$ pairs within the domain $[0, 1] \times [-1, 1]$. These points are used to enforce the PDE.

2.  **Optimization:** The `optax.adam` optimizer is used to update the network parameters. The training loop runs for a specified number of epochs, and the loss is printed periodically.

## Numerical Details

For our implementation, we used the following parameters:

*   Viscosity ($\nu$): $0.01 / \pi$
*   Neural Network Layers: `[2, 20, 20, 20, 20, 1]` (input layer, 4 hidden layers, output layer)
*   Epochs: 10,000
*   Number of Boundary/Initial Condition Points ($N_b$): 100
*   Number of Collocation Points ($N_c$): 10,000

## Results

After training, the PINN provides an approximation of the solution $u(t, x)$ over the entire domain. We can then visualize this solution. The generated `burgers_solution.png` shows the predicted velocity field, demonstrating how the shock wave develops and propagates over time.

![Predicted Burgers' Equation Solution](/assets/images/burgers_solution.png)

This approach highlights the power of PINNs in solving PDEs by leveraging automatic differentiation and neural network capabilities to embed physical laws directly into the learning process.
