# Tutorial 2: Building Deep Neural Networks

In the first tutorial, we built a single `Linear` layer to learn a straight line. But what if the data is not a straight line? What if we are trying to classify images of handwritten digits, or tell the difference between cats and dogs?

We need a **Deep Neural Network**.

## Non-Linear Data (The XOR Problem)

Imagine a dataset where you want to output `1` only if one of the inputs is `1`, but not both. This is the classic XOR (Exclusive OR) problem.

A single straight line *cannot* solve this. We need to bend and warp our mathematical space. To do this, we use **Activation Functions** like `ReLU`, `Sigmoid`, or `Tanh`.

```python
import numpy as np
from gradience import Tensor

# The XOR Dataset
X_data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# The labels (1 if exclusively one input is 1, else 0)
Y_data = np.array([
    [0],
    [1],
    [1],
    [0]
])

X = Tensor(X_data)
Y = Tensor(Y_data)
```

## Building a Multi-Layer Perceptron (MLP)

To solve this, we will build a 2-layer Neural Network. We can easily stack layers together using the `Sequential` container.

```python
from gradience.nn import Sequential, Linear, Tanh, Sigmoid

# We stack our layers in order
model = Sequential(
    # Hidden Layer: 2 inputs -> 16 neurons
    Linear(2, 16),
    # The Activation Function: This allows the network to bend!
    Tanh(),
    # Output Layer: 16 neurons -> 1 output
    Linear(16, 1),
    # Sigmoid squeezes the final output to be exactly between 0 and 1
    Sigmoid()
)
```

## Binary Classification

Because we are classifying whether the output is `0` or `1`, we are doing Binary Classification. Instead of `MSELoss` (Mean Squared Error), we should use `BCEWithLogitsLoss` or just use `MSELoss` since we applied `Sigmoid` manually. Let's stick with `MSELoss` for simplicity, but we will upgrade our optimizer to `Adam`!

```python
from gradience.nn.losses import MSELoss
from gradience.optim import Adam

criterion = MSELoss()

# Adam is a much smarter optimizer than standard SGD
optimizer = Adam(model.parameters(), lr=0.1)
```

## Training Loop

This is exactly identical to our Linear Regression loop! This is the beauty of Gradience: the training loop remains the same no matter how complex the model gets.

```python
epochs = 200

for epoch in range(epochs):
    # 1. Forward pass
    predictions = model(X)
    
    # 2. Calculate Loss
    loss = criterion(predictions, Y)
    
    # 3. Backward pass (calculate gradients)
    loss.backward()
    
    # 4. Update parameters
    optimizer.step()
    
    # 5. Clear gradients
    optimizer.zero_grad()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1} | Loss: {loss.item():.4f}")
```

## Making Predictions

After training, let's see what our model predicts!

```python
predictions = model(X)
print("Predictions (Raw):")
print(predictions.data)

# Round predictions to 0 or 1
rounded = np.round(predictions.data)
print("Predictions (Rounded):")
print(rounded)
```

You should see the network successfully predicting `[0, 1, 1, 0]`!

Congratulations, you have now trained a Deep Neural Network from scratch! You are now ready to tackle real-world datasets like MNIST. Check out the `examples/` directory in the repository to see exactly how to build an image classifier.
