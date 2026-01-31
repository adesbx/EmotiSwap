import cv2
import torch
import matplotlib.pyplot as plt
from cnnEmotion import CnnEmotion
from torchvision import transforms
from PIL import Image

def detectFace(path: str):
    img = cv2.imread(path)

    # print(img)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    # for (x, y, w, h) in face: # show the face with an rectangle
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # plt.figure(figsize=(20,10))
    # plt.imshow(img_rgb)
    # plt.axis('off')
    # plt.show()

    x, y, w, h = max(face, key=lambda f: f[2] * f[3])

    crop_img = img_rgb[y:y+h, x:x+w]
    crop_img = cv2.resize(crop_img, dsize=[48, 48])
    cv2.imwrite("./assets/img/test/crop.jpg", crop_img)

detectFace("./assets/img/test/sad.jpg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

idx_to_class = { # fait a la main a partir de l'ordre des dossiers dans assets/img/archive
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])


model = CnnEmotion()
model.load_state_dict(torch.load("./assets/models/model.pth", map_location=device))
model.to(device)
model.eval()

image = Image.open("./assets/img/test/crop.jpg").convert("RGB")
x = transform(image)
x = x.unsqueeze(0)  
x = x.to(device)

with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_class].item()

print(f"Classe prédite : {pred_class}")
print(f"Confiance : {confidence:.3f}")
print("Emotion :", idx_to_class[pred_class])
