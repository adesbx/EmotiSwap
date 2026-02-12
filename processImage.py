import cv2
import torch
import random
import json
import matplotlib.pyplot as plt
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat
from train import CnnEmotion
from torchvision import transforms
from PIL import Image
from pathlib import Path
from collections import deque

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

def detectFaceVideo(video_frame: cv2.typing.MatLike) -> cv2.typing.MatLike :
    gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    if len(face) == 0:
        return None
    
    img_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)

    # plt.figure(figsize=(20,10))
    # plt.imshow(img_rgb)
    # plt.axis('off')
    # plt.show()

    x, y, w, h = max(face, key=lambda f: f[2] * f[3])

    crop_img = img_rgb[y:y+h, x:x+w]
    crop_img = cv2.resize(crop_img, dsize=[48, 48])
    # plt.figure(figsize=(10,10))
    # plt.imshow(crop_img)
    # plt.axis('off')
    # plt.show()

    return crop_img

def randomImage(path: str):
    images = list(Path(path).glob("*"))
    images = [img for img in images if img.suffix.lower() in
              {".png", ".jpg", ".jpeg"}]
    return str(random.choice(images)) if images else None

def chooseAssociatedImage(emotion: str) -> str :
    match emotion:
        case "angry":
            return randomImage("./assets/img/imagesToSwap/angry")
        case "disgust":
            return randomImage("./assets/img/imagesToSwap/disgust")
        case "fear":
            return randomImage("./assets/img/imagesToSwap/fear")
        case "happy":
            return randomImage("./assets/img/imagesToSwap/happy")
        case "neutral":
            return randomImage("./assets/img/imagesToSwap/neutral")
        case "sad":
            return randomImage("./assets/img/imagesToSwap/sad")
        case "surprise":
            return randomImage("./assets/img/imagesToSwap/surprise")
        
def compareQueue(queue, actual_emotion, x) -> bool:
    if len(queue) >= x:
        copie_queue = deque()
        for _ in range(x):
            copie_queue.append(actual_emotion)
        
        if queue == copie_queue:
            return True
        
    return False

# detectFace(path="./assets/img/test/sad.jpg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("./assets/models/class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

model = CnnEmotion()
model.load_state_dict(torch.load("./assets/models/model.pth", map_location=device))
model.to(device)
model.eval()

video_capture = cv2.VideoCapture(0)
emotionQueue = deque()
successive_frame = 5
last_emotion = ""
with pyvirtualcam.Camera(width=1280, height=720, fps=30, fmt=PixelFormat.RGB) as cam:
    firstFrame = cv2.imread("./assets/img/imagesToSwap/neutral/neutral_cat_01.png") # trouver une meilleur manière # fait pour avoir une image de base (éviter le logo OBS)
    firstFrame = cv2.resize(firstFrame, dsize=[1280, 720])
    cam.send(firstFrame)
    cam.sleep_until_next_frame()

    while True:
        result, video_frame = video_capture.read()

        image = detectFaceVideo(video_frame)

        if result is False or image is None:
            
            default = cv2.imread("./assets/img/imagesToSwap/default_image.jpg") # image pas défaut
            default = cv2.resize(default, dsize=[1280, 720])
            default = cv2.cvtColor(default, cv2.COLOR_BGR2RGB)
            cam.send(default)
            cam.sleep_until_next_frame()
            continue

        image = Image.fromarray(image)
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

        if last_emotion != idx_to_class[pred_class]: #si émotion différente nouvelle image a afficher
            last_emotion = idx_to_class[pred_class]
            pathImgToShow = chooseAssociatedImage(idx_to_class[pred_class])
            imgToShow = cv2.imread(pathImgToShow)
            imgToShow = cv2.resize(imgToShow, dsize=[1280, 720])
            imgToShow = cv2.cvtColor(imgToShow, cv2.COLOR_BGR2RGB)

        if len(emotionQueue) >= successive_frame:
            emotionQueue.popleft()

        emotionQueue.append(idx_to_class[pred_class])

        if compareQueue(emotionQueue, idx_to_class[pred_class], successive_frame):
            cam.send(imgToShow)
            cam.sleep_until_next_frame()


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# video_capture.release()
# cv2.destroyAllWindows()