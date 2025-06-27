---
layout: post
title: "Manifolds, Functions, Functionals, and Their Neural Network Representations"
date: 2025-06-27 10:00:00 -0700
categories: [Mathematics, Neural Networks, Geometry, Functional Analysis]
---

## Introduction to Manifolds

In mathematics, a **manifold** is a topological space that locally resembles Euclidean space near each point. More formally, an $n$-dimensional manifold $M$ is a second-countable, Hausdorff topological space such that every point $p \in M$ has a neighborhood $U$ that is homeomorphic to an open subset of $\mathbb{R}^n$. This homeomorphism $\phi: U \to \mathbb{R}^n$ is called a **chart**, and the pair $(U, \phi)$ is a **coordinate chart**. A collection of charts that covers the entire manifold is called an **atlas**.

If the transition maps between overlapping charts are smooth (infinitely differentiable), the manifold is called a **smooth manifold**. Smooth manifolds are the setting for differential geometry and are crucial in physics for describing curved spaces like spacetime in general relativity, or phase spaces in classical mechanics.

**Example:** The surface of a sphere $S^2$ is a 2-dimensional manifold. Locally, any small patch on the sphere can be flattened out to resemble a piece of a 2D plane. However, no single chart can cover the entire sphere without singularities (e.g., mapping the poles to a single point).

## Functions on Manifolds

A **function on a manifold** $f: M \to \mathbb{R}$ is a mapping from the manifold to the real numbers. For a smooth manifold, we are particularly interested in **smooth functions**. A function $f: M \to \mathbb{R}$ is smooth if, for every chart $(U, \phi)$ in the atlas, the composite function $f \circ \phi^{-1}: \phi(U) \to \mathbb{R}$ is smooth in the usual Euclidean sense (i.e., it has continuous partial derivatives of all orders).

These functions can represent physical quantities that vary across a space, such as temperature distribution on a surface, or a scalar potential in a curved spacetime.

## Functionals on Manifolds

A **functional** is a mapping from a space of functions to the real numbers. In the context of manifolds, a functional $F: \mathcal{F}(M) \to \mathbb{R}$ takes a function $f \in \mathcal{F}(M)$ (where $\mathcal{F}(M)$ is some space of functions on the manifold $M$) and assigns a real number to it.

Functionals are ubiquitous in physics and engineering. For instance:

*   **Action Functional:** As seen in the previous post, the action $S[q(t)] = \int L(q, \dot{q}, t) dt$ is a functional that takes a path $q(t)$ (a function of time) and returns a scalar value.
*   **Energy Functional:** In elasticity or fluid dynamics, the total energy of a configuration (described by a function, e.g., displacement field) is often a functional.
*   **Integral of a Function:** The integral of a function over a manifold, e.g., $\int_M f \, dV$, is a functional that maps the function $f$ to a scalar value.

## Neural Network Representation of Functions on Manifolds

Representing a function $f: M \to \mathbb{R}$ using a neural network typically involves embedding the manifold into a higher-dimensional Euclidean space or using local coordinate charts.

**Approach 1: Global Embedding (if applicable):** If the manifold can be globally embedded in $\mathbb{R}^k$, then the neural network can take the coordinates in $\mathbb{R}^k$ as input.

**Approach 2: Local Charts (more general):** For a general manifold, a single neural network might not suffice. Instead, one could use a collection of neural networks, each associated with a chart. For a point $p \in M$, we find a chart $(U, \phi)$ containing $p$, compute its coordinates $\phi(p) \in \mathbb{R}^n$, and then feed these coordinates into a standard MLP $NN_U: \mathbb{R}^n \to \mathbb{R}$. The challenge here is ensuring consistency where charts overlap.

**Approach 3: Implicit Representation:** A single neural network can implicitly define the function. For example, if $M$ is a surface in $\mathbb{R}^3$, a neural network $NN: \mathbb{R}^3 \to \mathbb{R}$ could be trained such that $NN(x,y,z) = 0$ on the manifold, and then another network could represent the function on this implicitly defined manifold.

For a direct representation of $f: M \to \mathbb{R}$, where $M$ is an $n$-dimensional manifold, the neural network would take $n$ coordinates as input (from a chart) and output a single scalar value. The training data would consist of pairs $(\phi(p), f(p))$.

```python
import jax
import jax.numpy as jnp
from flax import linen as nn

class FunctionOnManifoldNN(nn.Module):
    features: list[int] # e.g., [64, 64, 1]
    input_dim: int # n-dimensional coordinates from a chart

    @nn.compact
    def __call__(self, coords): # coords is a 1D array of n coordinates
        x = coords.reshape(1, -1) # Ensure input is 2D for Dense layer
        for i, feat in enumerate(self.features[:-1]):
            x = nn.tanh(nn.Dense(feat, name=f"dense_{i}")(x))
        output_value = nn.Dense(self.features[-1], name="output_layer")(x)
        return output_value.squeeze() # Return scalar
```

## Neural Network Representation of Functionals on Manifolds

Representing a functional $F: \mathcal{F}(M) \to \mathbb{R}$ with a neural network is more complex, as the input to the neural network is itself a function (or a discretized representation of it).

**Approach 1: Discretization of Input Function:** This is the most common approach. The input function $f \in \mathcal{F}(M)$ is discretized into a finite set of values, similar to how we discretized paths for the action functional. If $f$ is defined on a manifold, we would sample $f$ at a finite number of points $p_1, p_2, \dots, p_K$ on the manifold. The neural network then takes the vector $(f(p_1), f(p_2), \dots, f(p_K))$ as input.

```python
class FunctionalNN(nn.Module):
    features: list[int] # e.g., [128, 64, 1]
    num_sample_points: int # K, the number of points where f is sampled

    @nn.compact
    def __call__(self, sampled_function_values): # 1D array of f(p_i) values
        x = sampled_function_values.reshape(1, -1)
        for i, feat in enumerate(self.features[:-1]):
            x = nn.tanh(nn.Dense(feat, name=f"dense_{i}")(x))
        functional_value = nn.Dense(self.features[-1], name="output_layer")(x)
        return functional_value.squeeze()
```

**Approach 2: Operator Learning (Neural Operators):** More advanced techniques, such as Neural Operators (e.g., Fourier Neural Operator, Graph Neural Operator), aim to learn the mapping between infinite-dimensional function spaces directly, without relying on a fixed discretization. These methods are particularly powerful for learning solutions to PDEs or other functional mappings.

**Training and Verification:**

Training such neural networks involves generating pairs of (input function, corresponding functional value). For example, if learning an energy functional, you would generate various configurations (functions) and compute their energies. The loss function would typically be the mean squared error between the predicted functional value and the true functional value.

Verifying the properties of these neural network representations (e.g., smoothness, invariance under certain transformations) often requires generating test data and checking the network's output against known mathematical properties. This is an active area of research at the intersection of machine learning and scientific computing.

