---
layout: post
title: "Galilean Invariance and Neural Network Representations of Action Functionals"
date: 2025-06-27 10:00:00 -0700
categories: [Physics, Mathematics, Neural Networks, Symmetry]
---

## The Profound Role of Symmetry in Physics

Symmetry is a cornerstone of modern physics, underpinning our understanding of fundamental laws and conservation principles. At its heart, a symmetry implies an invariance: a property of a system or its description that remains unchanged under certain transformations. In the context of classical mechanics and field theory, this often translates to the invariance of the **action functional**.

The action functional, $S$, is a central quantity in Lagrangian and Hamiltonian mechanics. For a system described by generalized coordinates $q(t)$, the action is typically an integral over time of the Lagrangian $L(q, \dot{q}, t)$:

$$ S = \int_{t_1}^{t_2} L(q(t), \dot{q}(t), t) \, dt $$

Physical laws are derived by applying the principle of least action, which states that the path taken by a system between two states is one for which the action is stationary (i.e., its variation is zero). A symmetry of the system then means that the action functional remains invariant under a particular transformation of the coordinates and time. Noether's theorem beautifully connects these continuous symmetries to conserved quantities.

## Galilean Invariance: A Simple Yet Powerful Example

Let's consider a simple system: a single free particle of mass $m$ moving in one dimension. Its Lagrangian is given by:

$$ L = \frac{1}{2} m \dot{x}^2 $$

And its action functional is:

$$ S = \int_{t_1}^{t_2} \frac{1}{2} m \dot{x}^2 \, dt $$

**Galilean invariance** states that the laws of classical mechanics are the same in all inertial frames of reference. An inertial frame is one that is either at rest or moving with constant velocity. Consider a transformation from one inertial frame $(x, t)$ to another $(x', t')$ moving with a constant velocity $v_0$ relative to the first:

$$ x' = x - v_0 t \\
 t' = t $$

To check the invariance of the action, we need to see how the Lagrangian transforms. First, let's find the transformed velocity:

$$ \dot{x}' = \frac{d}{dt'}(x - v_0 t) = \frac{d}{dt}(x - v_0 t) = \dot{x} - v_0 $$

Now, substitute $\dot{x}'$ into the Lagrangian in the primed frame:

$$ L' = \frac{1}{2} m (\dot{x}')^2 = \frac{1}{2} m (\dot{x} - v_0)^2 = \frac{1}{2} m (\dot{x}^2 - 2 v_0 \dot{x} + v_0^2) $$

$$ L' = \frac{1}{2} m \dot{x}^2 - m v_0 \dot{x} + \frac{1}{2} m v_0^2 $$

We can rewrite this as:

$$ L' = L - \frac{d}{dt}(m v_0 x) + \frac{1}{2} m v_0^2 $$

Now, let's look at the action in the primed frame:

$$ S' = \int_{t_1}^{t_2} L' \, dt = \int_{t_1}^{t_2} \left( L - \frac{d}{dt}(m v_0 x) + \frac{1}{2} m v_0^2 \right) \, dt $$

$$ S' = \int_{t_1}^{t_2} L \, dt - \int_{t_1}^{t_2} \frac{d}{dt}(m v_0 x) \, dt + \int_{t_1}^{t_2} \frac{1}{2} m v_0^2 \, dt $$

$$ S' = S - [m v_0 x]_{t_1}^{t_2} + \frac{1}{2} m v_0^2 (t_2 - t_1) $$

The terms $- [m v_0 x]_{t_1}^{t_2}$ and $+ \frac{1}{2} m v_0^2 (t_2 - t_1)$ are boundary terms and a constant term, respectively. Crucially, the equations of motion derived from the action are invariant under the addition of a total time derivative or a constant to the Lagrangian. Therefore, the action for a free particle is indeed Galilean invariant.

## Neural Network Representation of a Functional

Representing a functional with a neural network is a more advanced concept than representing a function. A functional maps a function (or a set of functions) to a scalar. In our case, the action functional $S[x(t)]$ maps a path $x(t)$ to a scalar value.

One approach to represent such a functional using neural networks is to discretize the path $x(t)$ into a sequence of points $(x_0, x_1, \dots, x_N)$ at discrete time steps $(t_0, t_1, \dots, t_N)$. The neural network would then take this sequence of points as input and output a scalar value representing the action.

Let's consider a simple neural network architecture for this:

*   **Input Layer:** Takes the discretized path $(x_0, x_1, \dots, x_N)$ as input. This could be a flattened vector of $N+1$ dimensions.
*   **Hidden Layers:** Standard fully connected layers (MLPs) to learn complex relationships within the path.
*   **Output Layer:** A single neuron with a linear activation function to output the scalar action value.

```python
import jax
import jax.numpy as jnp
from flax import linen as nn

class ActionFunctionalNN(nn.Module):
    features: list[int] # e.g., [64, 64, 1]
    num_time_steps: int

    @nn.compact
    def __call__(self, path_discretized): # path_discretized is a 1D array of x values
        x = path_discretized.reshape(1, -1) # Ensure input is 2D for Dense layer
        for i, feat in enumerate(self.features[:-1]):
            x = nn.tanh(nn.Dense(feat, name=f"dense_{i}")(x))
        action_value = nn.Dense(self.features[-1], name="output_layer")(x)
        return action_value.squeeze() # Return scalar

# Example usage:
# key = jax.random.PRNGKey(0)
# num_steps = 10
# nn_model = ActionFunctionalNN(features=[128, 64, 1], num_time_steps=num_steps)
# dummy_path = jnp.zeros(num_steps)
# params = nn_model.init(key, dummy_path)['params']
# print(nn_model.apply({'params': params}, dummy_path))
```

## Checking Galilean Invariance in a Neural Network Functional

To check if our neural network representation of the action functional is Galilean invariant, we need to perform the Galilean transformation on the input path and then observe if the output of the neural network changes by only a total time derivative or a constant.

Given a discretized path $x = (x_0, x_1, \dots, x_N)$ at times $t = (t_0, t_1, \dots, t_N)$, a Galilean transformation with velocity $v_0$ yields a new path $x' = (x'_0, x'_1, \dots, x'_N)$ where $x'_i = x_i - v_0 t_i$.

Let $S_{NN}(x)$ be the output of our neural network for a given path $x$. For Galilean invariance, we would ideally want:

$$ S_{NN}(x') = S_{NN}(x) - [m v_0 x]_{t_1}^{t_2} + \frac{1}{2} m v_0^2 (t_2 - t_1) $$

However, directly enforcing this during training can be challenging. Instead, we can incorporate this into the loss function during training. We would generate pairs of paths $(x, x')$ related by a Galilean transformation and add a term to the loss that penalizes deviations from this expected relationship.

**Proposed Loss Term for Galilean Invariance:**

Let $S_{NN}(x)$ be the action predicted by the neural network for path $x$, and $S_{NN}(x')$ for the transformed path $x'$. We can define a loss term $L_{Galilean}$ as:

$$ L_{Galilean} = \left( S_{NN}(x') - \left( S_{NN}(x) - (m v_0 x_N - m v_0 x_0) + \frac{1}{2} m v_0^2 (t_N - t_0) \right) \right)^2 $$

This term would be added to the overall training loss. The neural network would then be trained to minimize this loss, thereby learning to respect Galilean invariance. The challenge lies in generating diverse enough paths and transformations to ensure generalization.

**Numerical Verification:**

After training, to numerically verify Galilean invariance, you would:

1.  Generate a set of random paths $x(t)$.
2.  For each path, apply a range of Galilean transformations with different $v_0$ values to get $x'(t)$.
3.  Compute $S_{NN}(x)$ and $S_{NN}(x')$ using your trained neural network.
4.  Calculate the expected transformed action: $S_{expected} = S_{NN}(x) - (m v_0 x_N - m v_0 x_0) + \frac{1}{2} m v_0^2 (t_N - t_0)$.
5.  Check if $|S_{NN}(x') - S_{expected}|$ is close to zero (within a small tolerance).

This approach allows us to embed fundamental physical symmetries directly into the learning process of neural networks, leading to models that are not only data-driven but also physically consistent. This is a crucial step towards building more robust and interpretable AI models for scientific discovery.

<!-- Forced update to trigger Jekyll rebuild -->
