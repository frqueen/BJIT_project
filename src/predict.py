


import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

# import matplotlib.pyplot as plt

from model import *

def predict_image(model, image_path, device):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
    class_names = ['bike', 'car']
    prediction = class_names[predicted.item()]
    # # displaying the title
    # plt.title(prediction,
    #           fontsize='20',
    #           backgroundcolor='red',
    #           color='white')
    # plt.imshow(image)
    # plt.show()
    return prediction

# model = CarBikeClassifier(num_classes=2)
# model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
#
#
# image_file = '../data/test/Car/Car (1070).jpeg'
# pr=predict_image(model, image_file, device='cpu')
# print(pr)
#