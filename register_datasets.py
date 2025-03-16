from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances
import json
import os


def get_coco_dict(json_file, image_root):
    """
    Lit un fichier COCO JSON et renvoie une liste de dictionnaires au format Detectron2.

    Args:
        json_file (str): Chemin vers le fichier COCO JSON.
        image_root (str): Chemin vers le répertoire contenant les images.

    Returns:
        list[dict]: Liste de dictionnaires, où chaque dictionnaire représente une image
                   et ses annotations au format Detectron2.
    """
    with open(json_file, 'r') as f:
        coco_data = json.load(f)

    dataset_dicts = []
    for image_info in coco_data['images']:
        record = {}

        filename = os.path.join(image_root, image_info['file_name'])
        height = image_info['height']
        width = image_info['width']
        image_id = image_info['id']

        record["file_name"] = filename
        record["image_id"] = image_id
        record["height"] = height
        record["width"] = width

        annotations = []
        for ann in coco_data['annotations']:
            if ann['image_id'] == image_id:
                bbox = ann['bbox']

                annotation = {
                    "bbox": [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],  # Convertir en XYXY
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": ann['category_id'] - 1,  # Detectron2 indexe à partir de 0
                    "iscrowd": ann.get('iscrowd', 0)  # Par défaut 0 si absent
                }
                annotations.append(annotation)

        record["annotations"] = annotations
        dataset_dicts.append(record)

    return dataset_dicts


def register_coco_instances(name, json_file, image_root):
    """
    Enregistre un dataset COCO dans Detectron2.

    Args:
        name (str): Nom du dataset (ex: "my_dataset_train").
        json_file (str): Chemin vers le fichier COCO JSON.
        image_root (str): Chemin vers le répertoire contenant les images.
    """
    # Charger les métadonnées des catégories
    with open(json_file, 'r') as f:
        coco_data = json.load(f)

    categories = coco_data.get("categories", [])
    thing_classes = [c["name"] for c in categories]

    DatasetCatalog.register(name, lambda: get_coco_dict(json_file, image_root))
    MetadataCatalog.get(name).set(thing_classes=thing_classes, evaluator_type="coco")

    print(f"✅ Dataset '{name}' enregistré avec {len(thing_classes)} classes.")


# Définir les chemins
train_json = "C:/Users/Danielle/Desktop/stage_N3/dataset/split_data/train/train_coco.json"
train_image_root = "C:/Users/Danielle/Desktop/stage_N3/dataset/split_data/train/images"

val_json = "C:/Users/Danielle/Desktop/stage_N3/dataset/split_data/val/val_coco.json"
val_image_root = "C:/Users/Danielle/Desktop/stage_N3/dataset/split_data/val/images"

# Enregistrer les datasets
# Après enregistrement des datasets
register_coco_instances("my_dataset_train", train_json, train_image_root)
register_coco_instances("my_dataset_val", val_json, val_image_root)

# Vérification des métadonnées
metadata_train = MetadataCatalog.get("my_dataset_train")
metadata_train.thing_classes = ["Quadricoptere", "Hexacoptere", "Octocoptere", "Voilure"]
metadata_val = MetadataCatalog.get("my_dataset_val")
metadata_val.thing_classes = ["Quadricoptere", "Hexacoptere", "Octocoptere", "Voilure"]

print("📌 Métadonnées pour 'my_dataset_train' : ", metadata_train)
print("📌 Métadonnées pour 'my_dataset_val' : ", metadata_val)

if metadata_train is None:
    print("❌ Erreur : 'my_dataset_train' n'a pas été enregistré correctement.")
else:
    print("✅ 'my_dataset_train' enregistré correctement.")

if metadata_val is None:
    print("❌ Erreur : 'my_dataset_val' n'a pas été enregistré correctement.")
else:
    print("✅ 'my_dataset_val' enregistré correctement.")
