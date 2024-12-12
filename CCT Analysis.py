import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification
import seaborn as sns
import matplotlib.pyplot as plt

# Load the model architecture
# ViT Model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=10,
    ignore_mismatched_sizes=True
)

# Load the saved weights
checkpoint = torch.load("MangaVisualComparison/vit_checkpoint.pth", map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Define preprocessing steps
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class labels
class_labels = [
    "Akira Toriyama",
    "Boichi",
    "Eiichiro Oda",
    "Gege Akutami",
    "Koyoharu Gotouge",
    "Makoto Yukimura",
    "Tatsuki Fujimoto",
    "Tite Kubo",
    "Yukinobu Tatsu",
    "Yuuki Tabata"
]


# Function to perform inference
def predict(image_path, model):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class_idx = probabilities.argmax(dim=1).item()
        predicted_class_name = class_labels[predicted_class_idx]
        predicted_class_probability = probabilities[0, predicted_class_idx].item()
    return predicted_class_idx, predicted_class_name, predicted_class_probability, probabilities[0]


# Fan art image paths and assumed labels (-1 for unknown)
fan_art_paths = [
    "BoichiFanart/bf1.jpg", "BoichiFanart/bf2.jpg", "BoichiFanart/bf3.jpg",
    "BoichiFanart/bf4.jpg", "BoichiFanart/bf5.jpg", "BoichiFanart/bf6.jpg",
    "BoichiFanart/bf7.jpg", "BoichiFanart/bf8.jpg", "BoichiFanart/bf9.jpg",
    "BoichiFanart/bf10.jpg"
]
fan_art_true_labels = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  # Using -1 for unknown labels

# Real art image paths and correct labels (example indices)
real_art_paths = [
    "BoichiReal/br1.jpg", "BoichiReal/br2.jpg", "BoichiReal/br3.jpg",
    "BoichiReal/br4.jpg", "BoichiReal/br5.jpg", "BoichiReal/br6.jpg",
    "BoichiReal/br7.jpg", "BoichiReal/br8.png", "BoichiReal/br9.png",
    "BoichiReal/br10.jpg"
    ]
real_art_true_labels = [1] * 10

# Combine fan art and real art paths and labels
all_image_paths = fan_art_paths + real_art_paths
all_true_labels = fan_art_true_labels + real_art_true_labels

# Lists to store predictions and confidences
all_predictions = []
all_confidences = []

for image_path in all_image_paths:
    predicted_idx, predicted_name, predicted_prob, _ = predict(image_path, model)
    all_predictions.append(predicted_idx)
    all_confidences.append(predicted_prob)
    print(f"Image: {image_path}, Predicted: {predicted_name} ({predicted_prob * 100:.2f}%)")

# --- Analysis ---

# Separate indices for fan art and real art
fan_art_indices = [i for i, label in enumerate(all_true_labels) if label == -1]
real_art_indices = [i for i, label in enumerate(all_true_labels) if label != -1]

# Extract true labels and predictions for real art
real_true = [all_true_labels[i] for i in real_art_indices]
real_preds = [all_predictions[i] for i in real_art_indices]

# Compute metrics for real art
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(real_true, real_preds)
precision = precision_score(real_true, real_preds, average='weighted')
recall = recall_score(real_true, real_preds, average='weighted')
f1 = f1_score(real_true, real_preds, average='weighted')

print(f"\nMetrics for Real Art:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Confusion matrix for real art

cm = confusion_matrix(real_true, real_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Real Art')
plt.show()

# Confidence analysis for fan art
print("\nConfidence Scores for Fan Art:")
for i in fan_art_indices:
    print(f"{all_image_paths[i]}: {all_confidences[i] * 100:.2f}% confident in {class_labels[all_predictions[i]]}")
