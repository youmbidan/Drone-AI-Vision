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
from detectron2.structures import Instances, Boxes
from detectron2.data import Metadata

# --- Configuration des détecteurs ---
cfg_drone_types = get_cfg()
cfg_drone_types.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg_drone_types.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg_drone_types.MODEL.WEIGHTS = os.path.join("C:/Users/Danielle/Desktop/stage_N3/dataset/output/", "model_final.pth")
cfg_drone_types.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg_drone_types.MODEL.DEVICE = "cpu"
cfg_drone_types.OUTPUT_DIR = "C:/Users/Danielle/Desktop/stage_N3/dataset/output"
cfg_drone_types.freeze()

cfg_drone_defects = get_cfg()
cfg_drone_defects.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg_drone_defects.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg_drone_defects.MODEL.WEIGHTS = os.path.join("C:/Users/Danielle/Desktop/stage_N3/dataset/output2/", "model_final.pth")
cfg_drone_defects.MODEL.ROI_HEADS.NUM_CLASSES = 6
cfg_drone_defects.MODEL.DEVICE = "cpu"
cfg_drone_defects.OUTPUT_DIR = "C:/Users/Danielle/Desktop/stage_N3/dataset/output2"
cfg_drone_defects.freeze()

# --- Enregistrement des datasets ---
from detectron2.data.datasets import register_coco_instances

DATASET_ROOT = "C:/Users/Danielle/Desktop/stage_N3/dataset/split_data"
TRAIN_JSON = "C:/Users/Danielle/Desktop/stage_N3/dataset/split_data/train/train_coco.json"
TRAIN_IMG = "C:/Users/Danielle/Desktop/stage_N3/dataset/split_data/train/images"
VAL_JSON = "C:/Users/Danielle/Desktop/stage_N3/dataset/split_data/val/val_coco.json"
VAL_IMG = "C:/Users/Danielle/Desktop/stage_N3/dataset/split_data/val/images"
DRONE_TYPE_CLASS_NAMES = ["Quadricoptere", "Hexacoptere", "Octocoptere", "Voilure"]

DATASET_ROOT_DEFECTS = "C:/Users/Danielle/Desktop/stage_N3/dataset/split_data2"
TRAIN_JSON_DEFECTS = "C:/Users/Danielle/Desktop/stage_N3/dataset/split_data2/train/train2_coco.json"
TRAIN_IMG_DEFECTS = "C:/Users/Danielle/Desktop/stage_N3/dataset/split_data2/train/images"
VAL_JSON_DEFECTS = "C:/Users/Danielle/Desktop/stage_N3/dataset/split_data2/val/val2_coco.json"
VAL_IMG_DEFECTS = "C:/Users/Danielle/Desktop/stage_N3/dataset/split_data2/val/images"
DEFECT_CLASS_NAMES = ["Helice manquante/cassee", "Camera cassee", "Corps principal defectueux", "Pieds de support endommagés", "Drone detruit", "Batterie cassee/manquante"]

register_coco_instances("my_dataset_train", {}, TRAIN_JSON, TRAIN_IMG)
#MetadataCatalog.get("my_dataset_train").set(thing_classes=DRONE_TYPE_CLASS_NAMES) #Supprimer
register_coco_instances("my_dataset_val", {}, VAL_JSON, VAL_IMG)
#MetadataCatalog.get("my_dataset_val").set(thing_classes=DRONE_TYPE_CLASS_NAMES) #Supprimer

register_coco_instances("drone_defect_train", {}, TRAIN_JSON_DEFECTS, TRAIN_IMG_DEFECTS)
#MetadataCatalog.get("drone_defect_train").set(thing_classes=DEFECT_CLASS_NAMES) #Supprimer
register_coco_instances("drone_defect_val", {}, VAL_JSON_DEFECTS, VAL_IMG_DEFECTS)
#MetadataCatalog.get("drone_defect_val").set(thing_classes=DEFECT_CLASS_NAMES) #Supprimer

# --- Création des prédicteurs ---
predictor_drone_types = DefaultPredictor(cfg_drone_types)
predictor_drone_defects = DefaultPredictor(cfg_drone_defects)


def predict_and_visualize(image_path):
    """
    Effectue la détection de type de drone et d'anomalies et affiche les résultats combinés.
    """
    im = cv2.imread(image_path)

    # --- Prédictions séparées ---
    with torch.no_grad():
        outputs_drone_types = predictor_drone_types.predictor(im)
        instances_drone_types = outputs_drone_types["instances"].to("cpu")

        outputs_drone_defects = predictor_drone_defects.predictor(im)
        instances_drone_defects = outputs_drone_defects["instances"].to("cpu")

    # --- Combiner les instances ---
    combined_instances = Instances(im.shape[:2])

    # --- Combiner les champs des instances ---
    combined_instances.pred_classes = torch.cat((
        instances_drone_types.pred_classes if instances_drone_types.has("pred_classes") else torch.empty(0),
        instances_drone_defects.pred_classes + len(DRONE_TYPE_CLASS_NAMES) if instances_drone_defects.has("pred_classes") else torch.empty(0)
    ))

    combined_instances.scores = torch.cat((
        instances_drone_types.scores if instances_drone_types.has("scores") else torch.empty(0),
        instances_drone_defects.scores if instances_drone_defects.has("scores") else torch.empty(0)
    ))

    # Create Boxes objects if there are predictions
    boxes_drone_types = instances_drone_types.pred_boxes if instances_drone_types.has("pred_boxes") else None
    boxes_drone_defects = instances_drone_defects.pred_boxes if instances_drone_defects.has("pred_boxes") else None

    # Convert them to Boxes and then cat them correctly

    if boxes_drone_types is not None and boxes_drone_defects is not None:
         combined_instances.pred_boxes = Boxes.cat([boxes_drone_types, boxes_drone_defects])
    elif boxes_drone_types is not None:
        combined_instances.pred_boxes = boxes_drone_types
    elif boxes_drone_defects is not None:
        combined_instances.pred_boxes = boxes_drone_defects
    else:
        combined_instances.pred_boxes = None # or Boxes(torch.empty(0,4)) if empty boxes are needed

    # --- Créer des noms de classes combinés ---
    combined_class_names = DRONE_TYPE_CLASS_NAMES + DEFECT_CLASS_NAMES

    # Create a NEW Metadata instance
    metadata = Metadata()
    metadata.thing_classes = combined_class_names

    # --- Visualisation ---
    visualizer = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.8, instance_mode=ColorMode.IMAGE_BW)
    out = visualizer.draw_instance_predictions(combined_instances)

    # --- Affichage ---
    img_vis = out.get_image()[:, :, ::-1]

    # --- Redimensionnement de l'image pour l'affichage ---
    max_display_size = 800
    hauteur, largeur = img_vis.shape[:2]

    if largeur > max_display_size or hauteur > max_display_size:
        if largeur > hauteur:
            hauteur = int(hauteur * max_display_size / largeur)
            largeur = max_display_size
        else:
            largeur = int(largeur * max_display_size / hauteur)
            hauteur = max_display_size

        resized_img = cv2.resize(img_vis, (largeur, hauteur))
        cv2.imshow("Result", resized_img)
    else:
        cv2.imshow("Result", img_vis)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Define the correct predictor usage ---
class ModifiedDefaultPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.cfg.freeze()
        self.predictor = DefaultPredictor(self.cfg)

# Create the modified predictors
predictor_drone_types = ModifiedDefaultPredictor(cfg_drone_types)
predictor_drone_defects = ModifiedDefaultPredictor(cfg_drone_defects)
# --- Utilisation de la fonction de prédiction ---
image_path = "C:/Users/Danielle/Desktop/stage_N3/test1.jpg"  # Modifier avec le chemin de votre image
predict_and_visualize(image_path)