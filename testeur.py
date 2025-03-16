from datetime import datetime
import os
import json
import cv2
import torch
import multiprocessing
import io
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Image as ReportLabImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import Instances, Boxes
import register_datasets
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import Metadata

from metrics import cfg

setup_logger()

torch.set_num_threads(multiprocessing.cpu_count())
print("PyTorch est sur :", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def generate_report(image_paths):
    """Génère un rapport PDF après avoir analysé toutes les images."""
    try:
        # --- Configuration des détecteurs ---
        cfg_drone_types = get_cfg()
        cfg_drone_types.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg_drone_types.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg_drone_types.MODEL.WEIGHTS = os.path.join("C:/Users/Danielle/Desktop/stage_N3/dataset/output/", "model_final.pth")
        cfg_drone_types.MODEL.ROI_HEADS.NUM_CLASSES = 4
        cfg_drone_types.MODEL.DEVICE = "cpu"
        cfg_drone_types.freeze()

        cfg_drone_defects = get_cfg()
        cfg_drone_defects.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg_drone_defects.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg_drone_defects.MODEL.WEIGHTS = os.path.join("C:/Users/Danielle/Desktop/stage_N3/dataset/output2/", "model_final.pth")
        cfg_drone_defects.MODEL.ROI_HEADS.NUM_CLASSES = 6
        cfg_drone_defects.MODEL.DEVICE = "cpu"
        cfg_drone_defects.freeze()

        # --- Enregistrement des datasets ---

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


        # --- Création des prédicteurs ---
        predictor_drone_types = DefaultPredictor(cfg_drone_types)
        predictor_drone_defects = DefaultPredictor(cfg_drone_defects)

        # --- Fonctions de prédiction ---
        class ModifiedDefaultPredictor:
            def __init__(self, cfg):
                self.cfg = cfg.clone()
                self.cfg.freeze()
                self.predictor = DefaultPredictor(self.cfg)

        # Create the modified predictors
        predictor_drone_types = ModifiedDefaultPredictor(cfg_drone_types)
        predictor_drone_defects = ModifiedDefaultPredictor(cfg_drone_defects)

        def predictor_drone_types_func(image):
            """Effectue la prédiction des types de drones."""
            with torch.no_grad():
                outputs = predictor_drone_types.predictor(image)
            return outputs

        def predictor_drone_defects_func(image):
            """Effectue la prédiction des défauts des drones."""
            with torch.no_grad():
                outputs = predictor_drone_defects.predictor(image)
            return outputs

        # --- Initialisation du rapport PDF ---
        drone_definitions = {
            "Quadricoptere": "Un drone à quatre rotors, offrant stabilité et maniabilité.",
            "Hexacoptere": "Un drone à six rotors, offrant stabilité et maniabilité.",
            "Octocoptere": "Un drone à huit rotors, offrant stabilité et maniabilité.",
            "Voilure": "Un drone doté d'ailes fixes, offrant stabilité et maniabilité."
        }

        anomaly_recommendations = {
            "helicecassee/manquante": "Remplacer immédiatement l'hélice endommagée ou manquante.",
            "cameracassee": "Faire réparer ou remplacer la caméra endommagée.",
            "corpsprincipaldefectueux": "Inspecter et réparer ou remplacer la section principale endommagée du drone.",
            "dronedetruit": "Évaluer les dommages et déterminer si une réparation est possible ou si le remplacement est nécessaire.",
            "piedsdesupportendommages": "Remplacer les pieds de support endommagés pour assurer un atterrissage sûr.",
            "batteriecassee/manquante": "Remplacer la batterie endommagée ou manquante par une batterie en bon état."
        }

        # Styles PDF
        styles = getSampleStyleSheet()
        h1 = styles['Heading1']
        h1.fontName = 'Helvetica'
        h1.fontSize = 20
        h1.textColor = colors.royalblue

        h2 = styles['Heading2']
        h2.fontName = 'Helvetica'
        h2.fontSize = 16
        h2.textColor = colors.darkblue

        normal = styles["Normal"]
        normal.fontSize = 12
        normal.leading = 14

        # Création du document PDF
        cfg.OUTPUT_DIR = "C:/Users/Danielle/Desktop/stage_N3/dataset"
        report_path = os.path.join(cfg.OUTPUT_DIR, "report.pdf")

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        doc = SimpleDocTemplate(report_path, pagesize=letter)
        story = []

        # Ajouter page de Garde
        story.append(Paragraph("<b>Rapport d'Analyse de Drone</b>", h1))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(f"Date d'analyse: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal))
        story.append(Spacer(1, 1 * inch))
        story.append(Paragraph("<i>Ce rapport détaille les anomalies détectées sur le drone analysé.</i>", normal))
        story.append(PageBreak())

        # --- Fonctions pour effectuer la prédiction et générer la visualisation ---
        def process_image(image_path):
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
                combined_instances.pred_boxes = None  # or Boxes(torch.empty(0,4)) if empty boxes are needed

            # --- Créer des noms de classes combinés ---
            combined_class_names = DRONE_TYPE_CLASS_NAMES + DEFECT_CLASS_NAMES

            # Create a NEW Metadata instance
            metadata = Metadata()
            metadata.thing_classes = combined_class_names

            # --- Visualisation ---
            visualizer = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.8, instance_mode=ColorMode.IMAGE_BW)
            out = visualizer.draw_instance_predictions(combined_instances)

            return out.get_image()[:, :, ::-1], combined_instances, metadata

        # Traiter chaque image
        for image_path in image_paths:
            img_vis, combined_instances, metadata = process_image(image_path)

            # --- Section: Image et Détections ---
            story.append(Paragraph("<b>Image et Détections</b>", h2))
            story.append(Spacer(1, 0.1 * inch))

            # --- Convertir l'image visualisée en bytes pour ReportLab ---
            img_data = cv2.imencode('.png', img_vis)[1].tobytes()
            img = ReportLabImage(io.BytesIO(img_data), width=5 * inch, height=5 * inch)
            story.append(img)

            # --- Description de l'image ---
            story.append(Paragraph(f"Image analysée: {image_path}", normal))
            story.append(Spacer(1, 0.2 * inch))

            # --- Section: Détails des Détections ---
            story.append(Paragraph("<b>Détails des Détections</b>", h2))
            story.append(Spacer(1, 0.1 * inch))

            if len(combined_instances) > 0:
                for i in range(len(combined_instances)):
                    detected_class = metadata.thing_classes[combined_instances.pred_classes[i]]
                    score = combined_instances.scores[i].item()
                    story.append(Paragraph(f"<b>Détection:</b> {detected_class}", normal)) # Change ici
                    story.append(Paragraph(f"<b>Score de Confiance:</b> {score:.3f}", normal))

                    # --- Ajouter des recommandations basées sur le type de détection ---
                    if detected_class in drone_definitions:  # Si c'est un type de drone
                        recommendation = drone_definitions.get(detected_class, "Aucune recommandation spécifique définie.")
                        story.append(Paragraph(f"<b>Type de drone:</b> {recommendation}", normal))
                    elif detected_class in anomaly_recommendations:  # Si c'est une anomalie
                        recommendation = anomaly_recommendations.get(detected_class, "Aucune recommandation spécifique définie.")
                        story.append(Paragraph(f"<b>Recommandation:</b> {recommendation}", normal))
                    story.append(Spacer(1, 0.2 * inch))
            else:
                story.append(Paragraph("Aucun objet détecté dans l'image.", normal))
            story.append(PageBreak())

        # Générer le PDF
        doc.build(story)

        print(f"Rapport enregistré dans : {report_path}")
        return report_path

    except Exception as e:
        print(f"Erreur lors de la génération du rapport: {e}")
        return None