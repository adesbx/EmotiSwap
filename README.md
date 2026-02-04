# EmotiSwap

## À quoi sert EmotiSwap ?

EmotiSwap permet d’utiliser votre caméra afin de créer une caméra virtuelle temporaire qui diffuse une image en fonction de l’émotion détectée sur votre visage.

## Comment ça marche ?

Dans ce projet, j’ai entraîné un CNN simple à partir du dataset [FER2013](https://www.kaggle.com/datasets/msambare/fer2013?resource=download)

Ce dataset contient :
- 35 887 images
- réparties en 7 émotions

La détection du visage est réalisée à l’aide de haarcascade_frontalface_default.xml avec OpenCV.

L’image de votre visage est ensuite envoyée dans le modèle :
- une émotion est prédite
- une image correspondant à cette émotion est choisie aléatoirement
- cette image est diffusée via la caméra virtuelle

Les images utilisées peuvent être modifiées librement.

## Comment changer les images utilisées ?

Il suffit d’ajouter vos images dans les dossiers correspondant à chaque émotion :
```css
📁 /
 ┗ 📁 assets/
   ┗ 📁 img/
     ┗ 📁 imagesToSwap/
       ┣ 📁 angry/
       ┣ 📁 disgust/
       ┣ 📁 fear/
       ┣ 📁 happy/
       ┣ 📁 neutral/
       ┣ 📁 sad/
       ┗ 📁 surprise/
```
## Prérequis

Windows :
- OBS installé
- Caméra virtuelle OBS activée
- (OBS n’a pas besoin d’être lancé pour que le programme fonctionne)
---
Linux :
- Pas encore configuré
---

Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Exécution

Pour lancer le programme :
```bash
python processImage.py
```