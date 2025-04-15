import torch
import torch.nn.functional as F
from src.model import DenseCNN # Import the DenseCNN model definition from the model module


# Predict coordinates based on previous positions and current position.
def predict_coordinates(p2x, p2y, p1x, p1y, cx, cy, 
                       model_path="models/saved_models/best_model.pth"):
       # Initialize the model and load saved weights from model_path.
    model = DenseCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    

    # Create the input tensor by processing the given position values.
    input_tensor = create_input_tensor(p2x, p2y, p1x, p1y, cx, cy)
    

    # Make predictions without computing gradients.
    with torch.no_grad():
        # Pass the input tensor (unsqueezed to add batch dimension) through the model.
        outputs = model(input_tensor.unsqueeze(0))
        # Compute the softmax probabilities over the model outputs.
        probabilities = F.softmax(outputs, dim=1)
        # Get the top two prediction probabilities and corresponding indices.
        top_probs, top_indices = torch.topk(probabilities, 2, dim=1)
    

    # Retrieve the primary prediction index and its confidence.
    pred1_idx = top_indices[0, 0].item()
    pred1_conf = top_probs[0, 0].item()
    # Convert the flat index to y (row) and x (column) coordinates.
    pred1y, pred1x = divmod(pred1_idx, 10)
    

    # Retrieve the secondary prediction index and its confidence.
    pred2_idx = top_indices[0, 1].item()
    pred2_conf = top_probs[0, 1].item()
    # Convert the flat index to coordinates.
    pred2y, pred2x = divmod(pred2_idx, 10)
    
    # Return a tuple containing both predictions (x, y and confidence for primary and secondary)
    return (pred1x, pred1y, pred1_conf,
            pred2x, pred2y, pred2_conf)


# Helper function to create an input tensor from given position coordinates.
def create_input_tensor(p2x, p2y, p1x, p1y, cx, cy):
    # Inner function that creates a 10x10 image tensor filled with -1.0,
    # and sets the pixel at (x,y) to 1.0 if coordinates are valid.
    def _create_image(x, y):
        img = torch.full((10, 10), -1.0)
        if x >=0 and y >=0:
            img[y, x] = 1.0
        return img
    # Stack three images corresponding to p2, p1, and current positions.
    return torch.stack([
        _create_image(p2x, p2y),
        _create_image(p1x, p1y),
        _create_image(cx, cy)
    ])

# Main function to demonstrate prediction functionality.
def main():
    # Test prediction using sample coordinates
    results  = predict_coordinates(
        3,9,
        3,7,
        4,6
    )
    
    # Print formatted prediction results
    print("Predictions:")
    print(f"Primary: ({results[0]}, {results[1]}) - {results[2]*100:.1f}%")
    print(f"Secondary: ({results[3]}, {results[4]}) - {results[5]*100:.1f}%")


# Execute the main function when the script is run directly.
if __name__ == "__main__":
    main()