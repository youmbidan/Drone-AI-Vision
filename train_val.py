import json
import os
import random
import shutil  # Pour copier les images

def split_coco_dataset(coco_file, output_dir, image_dir):
    """
    Divise un fichier COCO JSON en ensembles d'entraînement et de validation (80/20)
    et copie les images correspondantes dans des dossiers séparés.

    Args:
        coco_file (str): Chemin vers le fichier COCO JSON combiné.
        output_dir (str): Répertoire de sortie pour les ensembles d'entraînement et de validation.
                          Les sous-dossiers 'train' et 'val' seront créés.
        image_dir (str): Répertoire où se trouvent les images référencées dans le fichier COCO JSON.
    """

    # Création des dossiers de sortie
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)

    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']

    # Créer un mappage d'image_id vers la liste des annotations associées
    image_annotation_map = {}
    for annotation in annotations:
        image_id = annotation['image_id']
        if image_id not in image_annotation_map:
            image_annotation_map[image_id] = []
        image_annotation_map[image_id].append(annotation)

    # Diviser les IDs d'images en ensembles d'entraînement et de validation (80/20)
    image_ids = list(image_annotation_map.keys()) #Utilise les ID d'images qui ont des annotations
    random.shuffle(image_ids) #Mélange les images

    train_image_ids = image_ids[:round(0.8 * len(image_ids))] #80% train
    val_image_ids = image_ids[round(0.8 * len(image_ids)):] #20% val

    # Crée les ensembles de données COCO pour l'entraînement et la validation
    train_coco = {'info': coco_data['info'], 'licenses': coco_data['licenses'], 'categories': categories, 'images': [], 'annotations': []}
    val_coco = {'info': coco_data['info'], 'licenses': coco_data['licenses'], 'categories': categories, 'images': [], 'annotations': []}

    # Remplir les ensembles d'entraînement et de validation
    for image_id in train_image_ids:
        image = next(img for img in images if img['id'] == image_id)
        train_coco['images'].append(image)

        # Copier l'image vers le dossier 'train'
        src_image_path = os.path.join(image_dir, image['file_name'])
        dst_image_path = os.path.join(train_dir, 'images', image['file_name'])
        shutil.copy(src_image_path, dst_image_path)

        # Ajouter les annotations associées à l'image
        train_coco['annotations'].extend(image_annotation_map[image_id])

    for image_id in val_image_ids:
        image = next(img for img in images if img['id'] == image_id)
        val_coco['images'].append(image)

        # Copier l'image vers le dossier 'val'
        src_image_path = os.path.join(image_dir, image['file_name'])
        dst_image_path = os.path.join(val_dir, 'images', image['file_name'])
        shutil.copy(src_image_path, dst_image_path)

        # Ajouter les annotations associées à l'image
        val_coco['annotations'].extend(image_annotation_map[image_id])

    # Supprimer les doublons d'annotations (très important si une même annotation est lié à une même image dans les différents fichier)
    train_coco['annotations'] = list({(ann['id']):ann for ann in train_coco['annotations']}.values())
    val_coco['annotations'] = list({(ann['id']):ann for ann in val_coco['annotations']}.values())

    # Enregistrer les fichiers COCO JSON pour l'entraînement et la validation
    with open(os.path.join(train_dir, 'train2_coco.json'), 'w') as f:
        json.dump(train_coco, f, indent=4)

    with open(os.path.join(val_dir, 'val2_coco.json'), 'w') as f:
        json.dump(val_coco, f, indent=4)

    print(f"Dataset split and saved to: {output_dir}")


# Utilisation
coco_file = "C:/Users/Danielle/Desktop/stage_N3/dataset/coco_fusionne.json"  # Fichier COCO JSON combiné
output_dir = "C:/Users/Danielle/Desktop/stage_N3/dataset/split_data2"  # Dossier de sortie pour les ensembles train et val
image_dir = "C:/Users/Danielle/Desktop/stage_N3/dataset/img_to"  # Dossier contenant toutes les images

split_coco_dataset(coco_file, output_dir, image_dir)