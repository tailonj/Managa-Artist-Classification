# Managa-Artist-Classification
The goal of this project is to explore the effectiveness of transformer-based models in identifying manga artists' unique styles. The project uses:

ViT (Vision Transformer) for its strong performance on image classification tasks.
CCT (Compact Convolutional Transformer) for its ability to generalize with smaller datasets.
The models are trained on 1000 images per artist and evaluated on a combination of manga panels and fan art to gauge their robustness and accuracy.

Usage
Cloning the Repository
To get started, clone the repository:

bash
git clone https://github.com/tailonj/Managa-Artist-Classification.git
cd Managa-Artist-Classification
Running Inference
For ViT Model:

python ViT\ Analysis.py

For CCT Model:

python CCT\ Analysis.py
Evaluation Data
Fan Art images are located in the Fanart directories.
Real Art images (manga panels) are located in the Real directories.

Key Functions
1. Inference Function
Performs model inference on a given image:

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
    
2. Metrics Calculation
Computes key evaluation metrics for real art classification:

Accuracy
Precision
Recall
F1 Score
Example:

accuracy = accuracy_score(real_true, real_preds)
precision = precision_score(real_true, real_preds, average='weighted')
recall = recall_score(real_true, real_preds, average='weighted')
f1 = f1_score(real_true, real_preds, average='weighted')

3. Confusion Matrix Visualization
Generates a confusion matrix to visualize model performance:

cm = confusion_matrix(real_true, real_preds)
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

Dependencies
Ensure the following libraries are installed:

pip install torch torchvision transformers scikit-learn seaborn matplotlib pillow
Setup and Installation
Create a Virtual Environment:

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install Dependencies:

pip install -r requirements.txt

Download Model Checkpoints:

Place the ViT and CCT model checkpoints in the appropriate directories:

/checkpoints/vit_checkpoint.pth
/checkpoints/cct_checkpoint.pth

Run the Analysis Scripts:

python ViT\ Analysis.py
python CCT\ Analysis.py

Results
Sample ViT Metrics
Accuracy: 60%
Precision: 100%
Recall: 60%
F1 Score: 75%

Sample CCT Metrics
Accuracy: 80%
Precision: 100%
Recall: 80%
F1 Score: 89%

Future Work
Data Diversification:

Include images from different sections of each manga to improve generalization.
Fan Art Fine-Tuning:

Fine-tune models on fan art to handle stylistic variations better.
Ensemble Models:

Combine ViT and CCT predictions to enhance overall accuracy.
Confidence Calibration:

Improve confidence estimation through techniques like temperature scaling.
License
This project is licensed under the MIT License.
