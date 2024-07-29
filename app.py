from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import torchvision.models as models

app = Flask(__name__)

# Loading the model
model = models.resnet18()
num_ftrs = model.fc.in_features
num_classes = 10 
model.fc = torch.nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load('cifar10_resnet18.pth'))
model.eval()

# Set device to GPU for faster processing
device = torch.device('cuda')
model = model.to(device)

# Define the transformation for the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# CIFAR-10 class names for printing out names after uploading images
class_names = {
    0: "Airplane",
    1: "Automobile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck"
}

def transforming(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

def predictioning(image_bytes):
    tensor = transforming(image_bytes).to(device)
    outputs = model(tensor)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image_bytes = file.read()
    class_id = predictioning(image_bytes)
    class_name = class_names[class_id]
    return jsonify({'class_id': class_id, 'class_name': class_name})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
