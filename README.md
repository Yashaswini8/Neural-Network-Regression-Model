# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
Regression is a supervised learning technique used to predict continuous values. Traditional regression models may fail to capture non-linear relationships in data. Neural Networks overcome this limitation by using multiple layers of neurons and non-linear activation functions.

A Neural Network Regression Model consists of:

An input layer that receives features

One or more hidden layers that learn complex patterns

An output layer that predicts a continuous value

During training, the model minimizes a loss function (Mean Squared Error) using backpropagation and an optimizer like Adam.

To build and train a neural network model using PyTorch to predict output values from input data and evaluate its performance using training loss and test data.

## Neural Network Model

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/80c3e4e3-316d-477d-af7a-8ca2fdc24aab" />

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: S.Yashaswini
### Register Number: 212224220123
```

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(ai_brain.parameters(), lr=0.01)

# Train (Reduced epochs so it runs fast)
epochs = 500
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = ai_brain(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# Plot
plt.plot(losses)
plt.title("Training Loss Vs Iteration")
plt.xlabel("Iterations")
plt.ylabel("Training Loss")
plt.show()

# Test
new_sample = torch.tensor([[35, 45]], dtype=torch.float32)
prediction = ai_brain(new_sample)

print("\nNew Sample Input: [35, 45]")
print("Predicted Output:", round(prediction.item(), 5))




```
## Dataset Information


<img width="1012" height="1054" alt="image" src="https://github.com/user-attachments/assets/e3f7457e-93e4-420f-8550-9618f8cc03c2" />


## OUTPUT

### Training Loss Vs Iteration Plot


<img width="922" height="664" alt="image" src="https://github.com/user-attachments/assets/df6ed088-7928-4296-9c45-f5ea99f86ff7" />

### New Sample Data Prediction



<img width="666" height="230" alt="image" src="https://github.com/user-attachments/assets/d01a89e8-11e6-49c8-ae24-bee66dd7f7d1" />

## RESULT

The program was executed using Command Prompt by placing the dataset and Python file in the same directory.
