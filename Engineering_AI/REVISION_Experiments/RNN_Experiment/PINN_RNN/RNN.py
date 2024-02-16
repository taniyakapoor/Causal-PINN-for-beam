# Importing Libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Defining sizes
input_size = 1
hidden_size = 32
output_size = 1
sequence_length = 100
batch_size = 1
num_epochs = 200

# Generate sine wave data
x_train = np.linspace(0, 2*np.pi, sequence_length+1)
y_train = np.sin(x_train)

# Split data into input and target sequences
input_data = y_train[:-1]
target_data = y_train[1:]

# Convert data to tensors
input_tensor = torch.tensor(input_data).view(batch_size, sequence_length, input_size).float()
target_tensor = torch.tensor(target_data).view(batch_size, sequence_length, output_size).float()

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output)
        return output, hidden

# Create RNN instance
rnn = RNN(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    # Set initial hidden state
    hidden = torch.zeros(1, batch_size, hidden_size)

    # Forward pass
    output, hidden = rnn(input_tensor, hidden)
    loss = criterion(output, target_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch+1) % 10 == 0:
        print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Generate predictions on new range of values
x_pred = np.linspace(0, 4*np.pi, 400)
input_pred = np.sin(x_pred)
input_tensor_pred = torch.tensor(input_pred[:-1]).view(batch_size, -1, input_size).float()
with torch.no_grad():
    hidden_pred = torch.zeros(1, batch_size, hidden_size)
    prediction, _ = rnn(input_tensor_pred, hidden_pred)

# Plot the results
plt.plot(x_train[:-1], input_data, label='Input Sequence (Train)')
plt.plot(x_train[1:], target_data, label='Target Sequence (Train)')
plt.plot(x_pred[:-1], prediction.view(-1).numpy(), label='Predicted Sequence (Prediction)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
