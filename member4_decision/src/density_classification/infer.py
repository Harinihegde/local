import torch
from PIL import Image
from torchvision import transforms

from config import DEVICE, IMAGENET_MEAN, IMAGENET_STD
from model import DensityClassifier

CLASS_NAMES = ['Sparse', 'Medium', 'Dense']

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def load_model(checkpoint_path='best_density_model.pth'):
    model = DensityClassifier(num_classes=3).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict_density(model, image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = val_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)
    return CLASS_NAMES[predicted.item()], confidence.item(), probs.cpu().numpy()[0]


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python infer.py <image_path>")
        sys.exit(1)
    model = load_model()
    pred_class, conf, probs = predict_density(model, sys.argv[1])
    print(f"Predicted: {pred_class} ({conf:.2%})")
    print(f"Probabilities: Sparse={probs[0]:.3f}, Medium={probs[1]:.3f}, Dense={probs[2]:.3f}")