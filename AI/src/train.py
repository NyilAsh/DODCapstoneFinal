import torch
from torch.utils.data import DataLoader
from model import DenseCNN
from dataset import PositionDataset
from utils.early_stopping import EarlyStopping

def main():
    # Configuration
    train_path = "data/data.csv"
    test_path = "data/testdata.csv"
    batch_size = 32
    learning_rate = 0.001
    max_epochs = 200
    patience = 10

    # Initialize components
    model = DenseCNN()
    train_data = PositionDataset(train_path)
    test_data = PositionDataset(test_path)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience)

    # Training loop
    for epoch in range(max_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss and accuracy
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == targets).sum().item()
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_correct += correct
            train_total += targets.size(0)
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                
                # Calculate validation metrics
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs.data, 1)
                
                val_loss += loss.item()
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)
        
        # Calculate percentages
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        # Early stopping check
        early_stopping(avg_val_loss, model)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{max_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}%\n")
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

if __name__ == "__main__":
    main()