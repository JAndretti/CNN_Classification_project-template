from .ImageProcessor import ImageProcessor
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor
import pandas as pd
from torchvision.io import read_image
from PIL import Image
import torch
from collections import defaultdict
import numpy as np


def data_process(input_path, output_path):
    processor = ImageProcessor(input_path)
    processor.remove_ds_store()
    print(processor.get_image_info())
    processor.images_to_tensor()
    # processor.augment_tensors()
    processor.normalize_tensors()
    processor.resize_tensors((100, 100))
    processor.save_tensors_as_images(output_path)


def to_one_hot(label, num_classes):
    """ Convertit une étiquette numérique en encodage one-hot. """
    one_hot = torch.zeros(num_classes)
    one_hot[label] = 1
    return one_hot


def load_pic(base_path):
    return ImageFolder(root=base_path, transform=ToTensor())

def organize_images(base_path, num_classes):
    """
    Organise les images selon leurs classes et renvoie des tenseurs et leurs étiquettes.

    :param base_path: Le chemin vers le dossier contenant les sous-dossiers de classes.
    :return: Un tuple (data, class_to_idx, paths) où chaque élément de data est un tuple (tensor, label)
             et le dictionnaire mappe les noms de classes aux indices.
             paths renvoie les chemins vers chaque image.
    """
    
    # Charger toutes les images et les convertir en tenseurs
    dataset = ImageFolder(root=base_path, transform=ToTensor())

    # Récupérer le dictionnaire de mapping classe vers index
    class_to_idx = dataset.class_to_idx

    # Extraire les chemins des images et les tenseurs
    paths = [sample[0] for sample in dataset.samples]
    data = [(tensor, to_one_hot(label,num_classes)) for tensor, label in dataset]

    # Compter le nombre d'images par classe
    class_counts = defaultdict(int)
    for _, label in data:
        label_idx = torch.argmax(label).item()
        class_name = [k for k, v in class_to_idx.items() if v == label_idx][0]
        class_counts[class_name] += 1

    # Imprimer les comptes
    print("Nombre d'images par classe:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")

    return data, class_to_idx, paths



def split_train_test(base_path, num_classes,test_size=0.3):
    """
    Sépare les images en ensembles de formation et de test et renvoie des tenseurs et leurs étiquettes.

    :param base_path: Le chemin vers le dossier contenant les sous-dossiers de classes.
    :param test_size: La proportion d'images à utiliser pour le test.
    :return: Un tuple (train_data, test_data, class_to_idx) où chaque élément est une liste de tuples (tensor, label) 
             et le dictionnaire mappe les noms de classes aux indices.
    """

    # Charger toutes les images et les convertir en tenseurs
    dataset = ImageFolder(root=base_path, transform=ToTensor())

    # Récupérer le dictionnaire de mapping classe vers index
    class_to_idx = dataset.class_to_idx

    # Shuffle and split indices for train and test
    indices = list(range(len(dataset)))
    split = int(np.floor(test_size * len(dataset)))
    
    np.random.shuffle(indices)
    
    train_indices, test_indices = indices[split:], indices[:split]
    
    train_paths = [dataset.samples[i][0] for i in train_indices]
    test_paths = [dataset.samples[i][0] for i in test_indices]
    
    train_data = [(dataset[i][0], to_one_hot(dataset[i][1],num_classes)) for i in train_indices]
    test_data = [(dataset[i][0], to_one_hot(dataset[i][1],num_classes)) for i in test_indices]

    # Compter le nombre d'images par classe dans train_data et test_data
    train_counts = defaultdict(int)
    test_counts = defaultdict(int)

    for _, label in train_data:
        label_idx = torch.argmax(label).item()
        class_name = [k for k, v in class_to_idx.items() if v == label_idx][0]
        train_counts[class_name] += 1

    for _, label in test_data:
        label_idx = torch.argmax(label).item()
        class_name = [k for k, v in class_to_idx.items() if v == label_idx][0]
        test_counts[class_name] += 1

    # Imprimer les comptes
    print("Nombre d'images par classe dans train_data:")
    for class_name, count in train_counts.items():
        print(f"{class_name}: {count}")

    print("\nNombre d'images par classe dans test_data:")
    for class_name, count in test_counts.items():
        print(f"{class_name}: {count}")

    return train_data, test_data, class_to_idx, train_paths, test_paths


def load_images_and_labels(csv_path, images_dir, class_to_num):
    """
    Charge les images et leurs labels depuis le chemin donné.

    :param csv_path: Chemin vers le fichier CSV contenant les noms des fichiers et leurs labels.
    :param images_dir: Chemin vers le dossier contenant les images.
    :param class_to_num: Dictionnaire convertissant les labels en nombres.
    :return: Liste de tuples contenant les tensors d'images et leurs labels.
    """
    # Lire le CSV pour obtenir les noms des fichiers et leurs labels
    df = pd.read_csv(csv_path)

    data = []
    for index, row in df.iterrows():
        # Modifiez l'extension si nécessaire
        image_name = "modified_" + str(row['id']).zfill(4) + ".jpg"
        image_path = os.path.join(images_dir, image_name)

        # Convertir l'image en tensor
        image_tensor = read_image(image_path).float()

        # Convertir le label en numéro
        label = class_to_num[row['label']]

        data.append((image_tensor, to_one_hot(label)))

    return data
