# src/predict.py
import torch
import torch.nn.functional as F
from model import DenseCNN

def predict_coordinates(input_data, model_path="models/saved_models/best_model.pth", top_k=100):
    """Predict coordinates with confidence scores, loading model automatically"""
    # Load model
    model = DenseCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Convert input to tensor if needed
    if not isinstance(input_data, torch.Tensor):
        input_data = create_input_tensor(input_data)
    
    with torch.no_grad():
        outputs = model(input_data.unsqueeze(0))  # Add batch dimension
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
    
    predictions = []
    for i in range(top_k):
        class_idx = top_indices[0, i].item()
        confidence = top_probs[0, i].item()
                
        y = class_idx // 10
        x = class_idx % 10
        pred = {'coordinates': (x, y), 'confidence': confidence}
        
        predictions.append(pred)
    
    return predictions

def create_input_tensor(data_dict):
    """Create input tensor from dictionary of coordinates"""
    def create_image(coords):
        img = torch.full((10, 10), -1.0)
        if coords['x'] >= 0 and coords['y'] >= 0:
            img[coords['y'], coords['x']] = 1.0
        return img
    
    return torch.stack([
        create_image(data_dict['p3']),
        create_image(data_dict['p2']),
        create_image(data_dict['p1'])
    ])

def main():
    # Test input data (format matches your CSV structure)
    test_data = {
        'p3': {'x': 2, 'y': 9},
        'p2': {'x': 2, 'y': 8},
        'p1': {'x': 2, 'y': 7}
    }
    
    # Get predictions
    predictions = predict_coordinates(test_data)
    
    # Display results
    print("\nPrediction Results:")
    print(f"Input Positions:")
    print(f"p3: {test_data['p3']['x']},{test_data['p3']['y']}")
    print(f"p2: {test_data['p2']['x']},{test_data['p2']['y']}")
    print(f"p1: {test_data['p1']['x']},{test_data['p1']['y']}\n")
    
    for i, pred in enumerate(predictions):
        print(f"Prediction {i+1}:")
        print(f"Coordinates: {pred['coordinates']}")
        print(f"Confidence: {pred['confidence']*100:.2f}%")
        print("-" * 40)

if __name__ == "__main__":
    main()