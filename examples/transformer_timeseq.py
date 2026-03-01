# This example demonstrates how to use a Transformer model for time series prediction
# Including data preparation, model construction, training and prediction

import riemann as rm
import riemann.nn as nn
from riemann.optim import Adam
from riemann.utils.data import Dataset, DataLoader
import numpy as np

# Generate time series data
def generate_time_series_data(num_samples, seq_length, pred_length):
    """
    Generate time series data
    
    :param num_samples: Number of samples
    :param seq_length: Input sequence length
    :param pred_length: Prediction sequence length
    :return: Input sequences and target sequences
    """
    # Generate sine wave data as an example
    t = np.linspace(0, 100, num_samples + seq_length + pred_length)
    data = np.sin(t) + 0.1 * np.random.randn(len(t))
    
    inputs = []
    targets = []
    
    for i in range(num_samples):
        inputs.append(data[i:i+seq_length])
        targets.append(data[i+seq_length:i+seq_length+pred_length])
    
    return np.array(inputs), np.array(targets)

# Custom time series dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=50, pred_length=10):
        self.inputs, self.targets = generate_time_series_data(
            num_samples, seq_length, pred_length
        )
        # Convert to Riemann tensors
        self.inputs = rm.tensor(self.inputs, dtype=rm.float32)
        self.targets = rm.tensor(self.targets, dtype=rm.float32)
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Simplified Transformer time series prediction model (encoder only)
class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, 
                    num_layers=2, dim_feedforward=128, pred_length=10):
        """
        Transformer time series prediction model (simplified version)
        
        Uses only Transformer encoder to map sequences to predictions
        
        :param input_dim: Input feature dimension
        :param d_model: Transformer model dimension
        :param nhead: Number of multi-head attention heads
        :param num_layers: Number of encoder layers
        :param dim_feedforward: Feedforward network dimension
        :param pred_length: Prediction sequence length
        """
        super(TransformerTimeSeriesModel, self).__init__()
        
        self.d_model = d_model
        self.pred_length = pred_length
        
        # Input embedding layer: map input dimension to d_model dimension
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding parameter (learnable positional encoding)
        self.pos_encoding = nn.Parameter(rm.randn(100, d_model) * 0.01)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Output layer: map d_model dimension to pred_length * input_dim
        self.output_layer = nn.Linear(d_model, pred_length * input_dim)
        
    def forward(self, src):
        """
        Forward pass
        
        :param src: Input sequence [batch_size, src_len, input_dim]
        :return: Prediction sequence [batch_size, pred_length, input_dim]
        """
        batch_size, src_len, input_dim = src.shape
        
        # Input embedding
        src = self.input_embedding(src)  # [batch_size, src_len, d_model]
        
        # Add positional encoding
        src = src + self.pos_encoding[:src_len, :].unsqueeze(0)
        
        # Encoder
        memory = self.transformer_encoder(src)  # [batch_size, src_len, d_model]
        
        # Take the last time step output
        last_output = memory[:, -1, :]  # [batch_size, d_model]
        
        # Output layer
        output = self.output_layer(last_output)  # [batch_size, pred_length * input_dim]
        
        # Reshape to [batch_size, pred_length, input_dim]
        output = output.view(batch_size, self.pred_length, input_dim)
        
        return output

# Create dataset and data loader
train_dataset = TimeSeriesDataset(num_samples=1000, seq_length=50, pred_length=10)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create model
model = TransformerTimeSeriesModel(
    input_dim=1, 
    d_model=64, 
    nhead=4, 
    num_layers=2,
    dim_feedforward=128,
    pred_length=10
)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Add feature dimension
        inputs = inputs.unsqueeze(-1)  # [batch_size, seq_len, 1]
        targets = targets.unsqueeze(-1)  # [batch_size, pred_len, 1]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

# Prediction example
model.eval()
with rm.no_grad():
    # Get a test sample
    test_input, test_target = train_dataset[0]
    test_input = test_input.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
    
    # Make prediction
    prediction = model(test_input)
    
    print(f"\nInput sequence shape: {test_input.shape}")
    print(f"Prediction sequence shape: {prediction.shape}")
    print(f"True target shape: {test_target.shape}")
    
    # Calculate prediction error
    test_target = test_target.unsqueeze(-1)
    error = rm.mean((prediction - test_target) ** 2).item()
    print(f"Prediction MSE: {error:.6f}")
    
    # Print target and prediction values
    print("\n===== Prediction Results Comparison =====")
    print(f"{'Step':<10} {'Target':<15} {'Prediction':<15} {'Error':<15}")
    print("-" * 55)
    
    pred_values = prediction.squeeze().tolist()
    target_values = test_target.squeeze().tolist()
    
    for i in range(len(target_values)):
        target_val = target_values[i]
        pred_val = pred_values[i] if isinstance(pred_values, list) else pred_values
        diff = target_val - pred_val
        print(f"{i+1:<10} {target_val:<15.6f} {pred_val:<15.6f} {diff:<15.6f}")
    
    print("-" * 55)