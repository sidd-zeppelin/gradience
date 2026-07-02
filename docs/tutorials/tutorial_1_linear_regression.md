# Tutorial 1: Training Your First Model

In this tutorial, we will learn how to use Gradience to train a simple machine learning model. We will teach the framework to recognize a simple mathematical pattern: $y = 2x + 1$. 

This is known as **Linear Regression**.

## Step 1: Prepare the Data

First, we need to create some examples so the model can learn. We will create inputs (`x`) and the correct answers (`y`).

```python
import numpy as np
from gradience import Tensor

# Create 100 random numbers between 0 and 1
x_data = np.random.rand(100, 1)

# Apply our secret formula: y = 2x + 1
# We also add a tiny bit of random noise to make it realistic
y_data = 2 * x_data + 1 + (np.random.randn(100, 1) * 0.1)

# Convert them into Gradience Tensors!
# We don't need gradients for data, so requires_grad=False (the default)
x = Tensor(x_data)
y = Tensor(y_data)
```

## Step 2: Build the Model

We will build our model by inheriting from the `Module` base class. 
Since our data has 1 feature (just `x`) and we want 1 output (just `y`), we will use a `Linear(1, 1)` layer.

```python
from gradience.nn import Module, Linear

class LinearRegressionModel(Module):
    def __init__(self):
        super().__init__()
        # A single fully-connected layer
        self.linear = Linear(in_features=1, out_features=1)
        
    def forward(self, x):
        # Pass the input through the linear layer
        return self.linear(x)

model = LinearRegressionModel()
```

## Step 3: Choose a Loss Function and Optimizer

The **Loss Function** measures how wrong our model is. For regression tasks, we use `MSELoss` (Mean Squared Error).
The **Optimizer** looks at the gradients and updates the weights. We will use standard Stochastic Gradient Descent (`SGD`).

```python
from gradience.nn.losses import MSELoss
from gradience.optim import SGD

criterion = MSELoss()
# Tell the optimizer to update our model's parameters, with a learning rate of 0.1
optimizer = SGD(model.parameters(), lr=0.1)
```

## Step 4: The Training Loop

Now for the fun part. We will loop over our data 100 times (100 "epochs"). In each loop, we will:
1. Make a prediction (`forward pass`).
2. Calculate the error (`loss`).
3. Calculate the gradients (`backward pass`).
4. Update the weights (`optimizer step`).
5. Clear the gradients for the next loop (`zero_grad`).

```python
epochs = 100

for epoch in range(epochs):
    # 1. Forward pass
    predictions = model(x)
    
    # 2. Calculate Loss
    loss = criterion(predictions, y)
    
    # 3. Backward pass (calculate gradients)
    loss.backward()
    
    # 4. Update parameters
    optimizer.step()
    
    # 5. Clear gradients
    optimizer.zero_grad()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1} | Loss: {loss.item():.4f}")
```

## Step 5: Check the Results!

Our model learned a linear formula: $y = Wx + B$. Let's look inside the `Linear` layer to see what Weight ($W$) and Bias ($B$) it discovered.

```python
weight = model.linear.weight.item()
bias = model.linear.bias.item()

print(f"Learned Formula: y = {weight:.2f}x + {bias:.2f}")
# Expected output: y = ~2.00x + ~1.00
```

Congratulations! You just built and trained your first AI model using Gradience from scratch. In the next tutorial, we'll dive into building Deep Neural Networks!
