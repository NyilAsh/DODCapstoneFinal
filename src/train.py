import torch
from torch.utils.data import DataLoader
from model import DenseCNN       # Import the DenseCNN model architecture
from dataset import PositionDataset  # Import the custom dataset for positions
from utils.early_stopping import EarlyStopping  # Import the EarlyStopping utility

def main():
    # Configuration parameters for training
    train_path = "data/data.csv"       # Path to training data CSV
    test_path = "data/testdata.csv"      # Path to testing data CSV
    batch_size = 32                    # Batch size for training and testing
    learning_rate = 0.001              # Learning rate for the optimizer
    max_epochs = 200                   # Maximum number of training epochs
    patience = 10                      # Early stopping patience

    # Initialize model and dataset instances
    model = DenseCNN()
    train_data = PositionDataset(train_path)
    test_data = PositionDataset(test_path)
    
    # Create data loaders for training and testing
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    # Setup optimizer and loss criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience)

    # Start the training loop over a fixed number of epochs
    for epoch in range(max_epochs):
        model.train()  # Set model to training mode
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        # Iterate over the batches in the training loader
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Clear gradients for current batch
            outputs = model(inputs)  # Forward pass through the model
            
            # Calculate loss between model outputs and actual targets
            loss = criterion(outputs, targets)
            # Obtain predictions from the model
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == targets).sum().item()
            
            # Backpropagation to compute gradients
            loss.backward()
            # Update model parameters based on computed gradients
            optimizer.step()
            
            # Update training metrics
            train_loss += loss.item()
            train_correct += correct
            train_total += targets.size(0)
        
        # After training, switch to evaluation mode to validate the model
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        # Evaluate without updating gradients
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                
                # Calculate loss and number of correct predictions on the validation set
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs.data, 1)
                
                val_loss += loss.item()
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)
        
        # Compute average losses and accuracy percentages
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        # Check early stopping condition with the average validation loss
        early_stopping(avg_val_loss, model)
        
        # Print current epoch statistics
        print(f"Epoch {epoch+1}/{max_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}%\n")
        
        # Break training loop if early stopping is triggered
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

# Execute main training function when the script is run directly.
if __name__ == "__main__":
    main()
