import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import os

import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode

# Configuration de base :
cfg = get_cfg()

# Ajouter la configuration du modèle à partir de Model Zoo :
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) #L'erreur provenais d'ici

# Mettre le seuil de confiance à 70 %
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

# Charger les poids d'un modèle entraîné localement
cfg.MODEL.WEIGHTS = os.path.join("C:/Users/Danielle/Desktop/stage_N3/dataset/output/", "model_final.pth")

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.MODEL.DEVICE = "cpu"

# Charger les méta-données :
# Enregistrez vos datasets en passant les métadonnées
from detectron2.data.datasets import register_coco_instances

# Chargement du métadonnée et utilisation de la librairie
cfg.MODEL.DEVICE = "cpu"
cfg.OUTPUT_DIR = "C:/Users/Danielle/Desktop/stage_N3/dataset/output"  # Repertoire

DATASET_ROOT = "C:/Users/Danielle/Desktop/stage_N3/dataset/split_data"

TRAIN_JSON = "C:/Users/Danielle/Desktop/stage_N3/dataset/split_data/train/train_coco.json"
TRAIN_IMG= "C:/Users/Danielle/Desktop/stage_N3/dataset/split_data/train/images"

VAL_JSON = "C:/Users/Danielle/Desktop/stage_N3/dataset/split_data/val/val_coco.json"
VAL_IMG = "C:/Users/Danielle/Desktop/stage_N3/dataset/split_data/val/images"
CLASS_NAMES = ["Quadricoptere", "Hexacoptere", "Octocoptere", "Voilure"] #Les classes a afficher

DatasetCatalog.register("my_dataset_train", lambda: DatasetCatalog(os.path.join(DATASET_ROOT, TRAIN_JSON), os.path.join(DATASET_ROOT, TRAIN_IMG))) #Load Train
MetadataCatalog.get("my_dataset_train").set(thing_classes=CLASS_NAMES) #Name

DatasetCatalog.register("my_dataset_val", lambda: DatasetCatalog(os.path.join(DATASET_ROOT, VAL_JSON), os.path.join(DATASET_ROOT, VAL_IMG))) #Load Train
MetadataCatalog.get("my_dataset_val").set(thing_classes=CLASS_NAMES) #Name
#Load et appliquer des infos sur le TRAIN DATA, ce qui facilite la génération futur et d'avoir des problèmes.

metadata = MetadataCatalog.get("my_dataset_train") #Load en mémoire

# Load weights from training output
# 3. Création du prédicteur:
predictor = DefaultPredictor(cfg) #Lance une configuration !

# Charger l'image:
image_path = "C:/Users/Danielle/Desktop/stage_N3/test.jpg" #Modifier avec chemin
im = cv2.imread(image_path) #Lecture avec CV2
print (image_path)

# Prédiction :
with torch.no_grad(): #Desactive le calcul du gradient
    outputs = predictor(im)

# Visualisation:
v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.8, instance_mode=ColorMode.IMAGE_BW)
out = v.draw_instance_predictions(outputs["instances"].to("cpu")) #Prédiction!
hauteur, largeur = out.get_image().shape[:2] #Recuperation de la hauteur et de la largeur

#Afficher les information de l'image
print(f"Originale - Largeur: {largeur}, Hauteur: {hauteur}")

# Définir une taille d'image maximale
max_display_size = 800

# Redimensionner l'image de sortie
resized_img = out.get_image()[:, :, ::-1] # Initialiser resized_img avec l'image originale

if largeur > max_display_size or hauteur > max_display_size:
    if largeur > hauteur:
        hauteur = int(hauteur * max_display_size / largeur)
        largeur = max_display_size
    else:
        largeur = int(largeur * max_display_size / hauteur)
        hauteur = max_display_size

    resized_img = cv2.resize(out.get_image()[:, :, ::-1], (largeur, hauteur))  # Redimensionner le visuel de sortie
    print(f"Redimensionnée - Largeur: {largeur}, Hauteur: {hauteur}")

#Afficher l'image résultante avec CV2
cv2.imshow("Result", resized_img)

cv2.waitKey(0)  # Attendre jusqu'à ce qu'une touche soit pressée
cv2.destroyAllWindows()  # Fermer toutes les fenêtres