from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from src import cct_14_7x2_224
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification

app = Flask(__name__)

# Define artist mapping
ARTIST_MAPPING = {
    0: "Akira Toriyama",
    1: "Boichi",
    2: "Eiichiro Oda",
    3: "Gege Akutami",
    4: "Koyoharu Gotouge",
    5: "Makoto Yukimura",
    6: "Tatsuki Fujimoto",
    7: "Tite Kubo",
    8: "Yukinobu Tatsu",
    9: "Yuuki Tabata"
}

# Load models globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CCT Model
cct_model = cct_14_7x2_224(pretrained=False, num_classes=10)
cct_checkpoint = torch.load("mangart_checkpoint.pth", map_location='cpu')
cct_model.load_state_dict(cct_checkpoint['model_state_dict'])
cct_model.to(device)
cct_model.eval()

# ViT Model
vit_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=10,
    ignore_mismatched_sizes=True
)
vit_checkpoint = torch.load("vit_checkpoint.pth", map_location='cpu')
vit_model.load_state_dict(vit_checkpoint['model_state_dict'])
vit_model.to(device)
vit_model.eval()

# Preprocessing function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.route('/')
def index(): # Serve the frontend.
    return render_template('index.html')


# Perform inference and return predictions.
def predict(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        probabilities = F.softmax(logits, dim=1)
        predicted_class_idx = probabilities.argmax(dim=1).item()

        return {
            "predicted_class": ARTIST_MAPPING[predicted_class_idx],
            "confidence": f"{probabilities[0, predicted_class_idx].item() * 100:.2f}%"
        }


@app.route('/cct', methods=['POST'])
def cct_inference():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']
    input_tensor = preprocess_image(image)
    predictions = predict(cct_model, input_tensor)
    return jsonify(predictions)


@app.route('/vit', methods=['POST'])
def vit_inference():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']
    input_tensor = preprocess_image(image)
    predictions = predict(vit_model, input_tensor)
    return jsonify(predictions)


def preprocess_image(image):
    """Preprocess the input image."""
    image = Image.open(image).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


if __name__ == '__main__':
    app.run(debug=True)
