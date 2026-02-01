import cv2
import torch
import random
import matplotlib.pyplot as plt
from cnnEmotion import CnnEmotion
from torchvision import transforms
from PIL import Image
from pathlib import Path

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

    # for (x, y, w, h) in face: # show the face with an rectangle
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

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

# detectFace(path="./assets/img/test/sad.jpg")
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

video_capture = cv2.VideoCapture(0)

while True:
    result, video_frame = video_capture.read()

    if result is False:
        break

    image = detectFaceVideo(video_frame)
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

    pathImgToShow = chooseAssociatedImage(idx_to_class[pred_class])
    imgToShow = cv2.imread(pathImgToShow)
    plt.figure(figsize=(10,10))
    plt.imshow(imgToShow)
    plt.axis('off')
    plt.text(
    0, 0,              
    idx_to_class[pred_class],           
    color="blue",
    fontsize=20,
    bbox=dict(facecolor="black", alpha=0.5)
)
    plt.show()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()