import pandas as pd
from tqdm import tqdm
from PIL import Image
import os
import torch
from torchvision.io import read_image, write_jpeg
from torchvision.utils import save_image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import (Compose, Resize, ToPILImage, ToTensor,
                                    Normalize, Grayscale, RandomRotation, RandomAffine,
                                    RandomResizedCrop, RandomHorizontalFlip,
                                    ColorJitter, GaussianBlur)


class ImageProcessor:
    def __init__(self, path):
        self.path = path

    # Fonction pour supprimer le fichier .DS_Store s'il existe
    def remove_ds_store(self):
        for dirpath, dirnames, filenames in os.walk(self.path):
            for filename in filenames:
                if filename.endswith(".DS_Store"):
                    ds_store_path = os.path.join(dirpath, filename)
                    os.remove(ds_store_path)

    def images_to_tensor(self):
        tensors_list = []
        paths_list = []

        # Parcourir tous les dossiers et sous-dossiers
        for dirpath, dirnames, filenames in os.walk(self.path):
            for filename in filenames:
                # Supprimer .DS_Store si vous ne l'avez pas déjà fait
                if filename == ".DS_Store":
                    continue

                image_path = os.path.join(dirpath, filename)
                paths_list.append(image_path)

                # Ouvrir l'image et la convertir en tenseur
                with Image.open(image_path) as img:
                    tensor = ToTensor()(img) * 255
                    tensors_list.append(tensor)

        self.tensors = tensors_list
        self.image_paths = paths_list

    def get_image_info(self):
        class_count = 0
        total_images = 0
        images_per_class = {}
        size_set = set()

        # Parcourir tous les sous-dossiers
        for dirpath, dirnames, filenames in os.walk(self.path):
            # Ignorer le dossier racine
            if dirpath == self.path:
                continue

            # Ignorer les fichiers .DS_Store
            filenames = [f for f in filenames if f != ".DS_Store"]

            # Compter le nombre de classes
            class_count += 1

            # Compter le nombre d'images par classe
            images_per_class[dirpath.split('/')[-1]] = len(filenames)

            # Compter le nombre total d'images
            total_images += len(filenames)

            # Vérifier les dimensions des images
            for filename in filenames:
                image_path = os.path.join(dirpath, filename)
                with Image.open(image_path) as img:
                    size_set.add(img.size)

        same_size = len(size_set) == 1

        return {
            'total_images': total_images,
            'number_of_classes': class_count,
            'images_per_class': images_per_class,
            'same_size_for_all_images': same_size
        }

    def normalize_tensors(self):
        self.tensors = [tensor / 255.0 for tensor in self.tensors]

    def compute_mean_std(self):
        # Concaténer tous les tensors le long d'une nouvelle dimension
        all_tensors = torch.stack(self.tensors, dim=0)

        # Calculer la moyenne et l'écart type pour chaque canal
        mean = torch.mean(all_tensors, dim=(0, 2, 3))
        std = torch.std(all_tensors, dim=(0, 2, 3))

        return mean, std

    def standardize_tensors(self):
        mean, std = self.compute_mean_std()

        # Redimensionner mean et std pour qu'ils soient de forme [3, 1, 1]
        mean = mean[:, None, None]
        std = std[:, None, None]

        standardized_tensors = [
            (tensor - mean) / std for tensor in self.tensors]

        self.tensors_standardize = standardized_tensors

    def resize_tensors(self, new_size=(90, 90)):
        resized_tensors = []

        for tensor in self.tensors:
            # Convertir le tenseur en image PIL pour utiliser la transformation `Resize`
            pil_image = ToPILImage()(tensor)

            # Redimensionner l'image
            resized_image = Resize(new_size)(pil_image)

            # Convertir l'image redimensionnée en tenseur
            resized_tensor = ToTensor()(resized_image)
            resized_tensors.append(resized_tensor)

        self.tensors = resized_tensors

    def augment_tensors(self):
        # Définir les augmentations
        augmentations = Compose([
            RandomRotation(30),                           # Rotation aléatoire
            RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translation
            # Redimensionnement aléatoire (zoom)
            RandomResizedCrop(224, scale=(0.8, 1.0)),
            # Miroir (flip) horizontalement
            RandomHorizontalFlip(p=0.5),
            # Modification de la luminosité, contraste, saturation
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))  # Bruitage
        ])

        augmented_tensors = []

        for tensor in tqdm(self.tensors):
            # Convertir le tenseur en image PIL pour utiliser les augmentations
            pil_image = ToPILImage()(tensor)

            # Appliquer les augmentations
            augmented_image = augmentations(pil_image)

            # Convertir l'image augmentée en tenseur
            augmented_tensor = ToTensor()(augmented_image)
            augmented_tensors.append(augmented_tensor)

        self.tensors = augmented_tensors

    def convert_to_grayscale(self):
        grayscale_tensors = []

        for tensor in self.tensors:
            # Convertir le tenseur en image PIL pour utiliser la transformation `Grayscale`
            pil_image = ToPILImage()(tensor)

            # Convertir l'image en nuances de gris
            grayscale_image = Grayscale(num_output_channels=1)(pil_image)

            # Convertir l'image en nuances de gris en tenseur
            grayscale_tensor = ToTensor()(grayscale_image)
            grayscale_tensors.append(grayscale_tensor)

        self.tensors = grayscale_tensors

    def save_tensors_as_images(self, output_dir, prefix="modified_"):
        for tensor, image_path in tqdm(zip(self.tensors, self.image_paths)):
            class_name = os.path.basename(os.path.dirname(image_path))
            new_dir = os.path.join(output_dir, class_name)

            # Créer un sous-dossier pour la classe s'il n'existe pas
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            # Construire le nouveau chemin d'image
            new_image_name = prefix + os.path.basename(image_path)
            new_image_path = os.path.join(new_dir, new_image_name)

            # Enregistrer le tenseur sous forme d'image
            save_image(tensor, new_image_path)

    @staticmethod
    def display_tensor_as_image(tensor):
        """
        Affiche un tensor comme une image.

        Args:
        - tensor (torch.Tensor): Un tensor de taille 3x50x50 avec des valeurs entre 0 et 1.
        """
        # Vérification des dimensions et des valeurs du tensor
        assert tensor.shape == (
            3, 50, 50), "Le tensor doit avoir la forme (3, 50, 50)"
        assert torch.max(tensor) <= 1.0 and torch.min(
            tensor) >= 0.0, "Les valeurs du tensor doivent être entre 0 et 1"

        # Transpose le tensor pour avoir les dimensions 50x50x3
        image_data = tensor.permute(1, 2, 0).numpy()

        # Affiche l'image
        plt.imshow(image_data)
        plt.axis('off')  # Pour désactiver les axes
        plt.show()

    @staticmethod
    def tensor_info(tensor):
        """
        Retourne le nombre de valeurs uniques et la plage de valeurs d'un tensor.

        Args:
        - tensor (torch.Tensor): Un tensor de taille 3x100x100.

        Returns:
        - int: Nombre de valeurs uniques.
        - tuple: Plage des valeurs (min, max).
        """
        # Vérification des dimensions du tensor
        assert tensor.shape == (
            3, 100, 100), "Le tensor doit avoir la forme (3, 100, 100)"

        # Compte le nombre de valeurs uniques
        unique_values = torch.unique(tensor)
        num_unique_values = len(unique_values)

        # Trouve le minimum et le maximum du tensor
        min_val = torch.min(tensor).item()
        max_val = torch.max(tensor).item()

        return num_unique_values, (min_val, max_val)
