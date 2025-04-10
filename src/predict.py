import torch
import torch.nn.functional as F
from model import DenseCNN

def predict_coordinates(p3x, p3y, p2x, p2y, p1x, p1y, 
                       model_path="models/saved_models/best_model.pth"):
    model = DenseCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    input_tensor = create_input_tensor(p3x, p3y, p2x, p2y, p1x, p1y)
    
    with torch.no_grad():
        outputs = model(input_tensor.unsqueeze(0))
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, 2, dim=1)
    
    pred1_idx = top_indices[0, 0].item()
    pred1_conf = top_probs[0, 0].item()
    pred1y, pred1x = divmod(pred1_idx, 10)
    
    pred2_idx = top_indices[0, 1].item()
    pred2_conf = top_probs[0, 1].item()
    pred2y, pred2x = divmod(pred2_idx, 10)
    
    return (pred1x, pred1y, pred1_conf,
            pred2x, pred2y, pred2_conf)

def create_input_tensor(p3x, p3y, p2x, p2y, p1x, p1y):
    def _create_image(x, y):
        img = torch.full((10, 10), -1.0)
        if x >=0 and y >=0:
            img[y, x] = 1.0
        return img
    return torch.stack([
        _create_image(p3x, p3y),
        _create_image(p2x, p2y),
        _create_image(p1x, p1y)
    ])

def main():
    results  = predict_coordinates(
        3,9,
        3,7,
        4,6
    )
    
    print("Predictions:")
    print(f"Primary: ({results[0]}, {results[1]}) - {results[2]*100:.1f}%")
    print(f"Secondary: ({results[3]}, {results[4]}) - {results[5]*100:.1f}%")

if __name__ == "__main__":
    main()