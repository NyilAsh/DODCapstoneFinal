# src/predict.py
import torch
import torch.nn.functional as F
from model import DenseCNN

def predict_coordinates(p3x, p3y, p2x, p2y, p1x, p1y, 
                       model_path="models/saved_models/best_model.pth", 
                       top_k=2):
    """Predict coordinates from 6 input integers"""
    # Load model
    model = DenseCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Create input tensor
    input_tensor = create_input_tensor(p3x, p3y, p2x, p2y, p1x, p1y)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_tensor.unsqueeze(0))
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
    
    # Convert to coordinates
    predictions = []
    for i in range(top_k):
        class_idx = top_indices[0, i].item()
        confidence = top_probs[0, i].item()
        y = class_idx // 10
        x = class_idx % 10
        predictions.append({'coordinates': (x, y), 'confidence': confidence})
    
    return predictions

def create_input_tensor(p3x, p3y, p2x, p2y, p1x, p1y):
    """Create input tensor from 6 coordinate values"""
    def _create_image(x, y):
        img = torch.full((10, 10), -1.0)
        if x >= 0 and y >= 0:
            img[y, x] = 1.0
        return img
    
    return torch.stack([
        _create_image(p3x, p3y),  # p3 image
        _create_image(p2x, p2y),  # p2 image
        _create_image(p1x, p1y)   # p1 image
    ])

def main():
    # Example usage with 6 input coordinates
    predictions = predict_coordinates(
        3,9,  # No p3 position
        3,7,
        4,6
    )
    
    print("Top Predictions:")
    for i, pred in enumerate(predictions):
        print(f"{i+1}. {pred['coordinates']} ({pred['confidence']*100:.1f}% confidence)")

if __name__ == "__main__":
    main()