# Library (Lib)

## Présentation

La bibliothèque `Lib` est destinée à faciliter la préparation et le traitement d'images pour les applications de machine learning, en particulier avec PyTorch. Elle contient plusieurs modules clés : `ImageProcessor`, `cnn.py`, et `data_prep.py`.

## Modules

## ImageProcessor Class (ImageProcessor.py)

La classe `ImageProcessor` est conçue pour faciliter la gestion, la transformation et l'augmentation d'images contenues dans un dossier spécifique.

## Présentation

La classe `ImageProcessor` est conçue pour faciliter la gestion, la transformation et l'augmentation d'images contenues dans un dossier spécifique. Elle offre des fonctionnalités essentielles pour le prétraitement d'images en vue de leur utilisation dans le contexte du machine learning, en particulier avec PyTorch.

## Fonctionnalités

- **Suppression de .DS_Store** : Retire les fichiers indésirables générés sur macOS.
- **Conversion en Tenseurs** : Transforme toutes les images du dossier en une liste de tenseurs PyTorch.
- **Normalisation** : Ajuste les tenseurs pour qu'ils aient des valeurs comprises entre 0 et 1.
- **Redimensionnement** : Change la taille de toutes les images pour qu'elles correspondent à une dimension donnée.
- **Augmentation** : Applique un ensemble de transformations pour augmenter la diversité du dataset.
- **Nuances de Gris** : Convertit les images RGB en nuances de gris.
- **Informations sur les Images** : Renvoie des informations comme le nombre total d'images, le nombre de classes, et si toutes les images ont les mêmes dimensions.
- **Standardisation des Tenseurs** : Standardise les tenseurs en utilisant leur moyenne et écart type.

## Utilisation

### Initialisation

Pour créer une instance de la classe :
```python
processor = ImageProcessor('chemin_vers_les_images')
```

### Traitement

- Suppression des fichiers .DS_Store :
```python
processor.remove_ds_store()
```
- Conversion des images en tenseurs :
```python
processor.images_to_tensor()
```
- Redimensionnement des images :
```python
processor.resize_tensors(128, 128)
```
- Augmentation des images :
```python
processor.augment_tensors()
```
- Conversion en nuances de gris :
```python
processor.convert_to_grayscale()
```
- Informations sur les images :
```python
info = processor.get_image_info()
```
- Calcul de la moyenne et de l'écart type :
```python
mean, std = processor.compute_mean_std()
```
- Standardisation des tenseurs :
```python
processor.standardize_tensors()
```
- Sauvegarder les tenseurs comme images :
```python
processor.save_tensors_as_images('chemin_du_dossier_de_sortie')
```

### Utilitaires

- Afficher un tensor comme une image :
```python
ImageProcessor.display_tensor_as_image(tensor)
```
- Obtenir des informations sur un tensor :
```python
unique_values, value_range = ImageProcessor.tensor_info(tensor)
```

### Accès aux Tenseurs

Pour accéder à la liste des tenseurs :
```python
tensors = processor.tensors
```

## data_prep.py

Ce module contient des fonctions destinées à la préparation et à la manipulation de données d'images. Il utilise `ImageProcessor` ainsi que d'autres fonctions pour faciliter le traitement des images.

**Fonctions principales :**

- `data_process(input_path, output_path)` : Applique une série de traitements aux images à partir d'un chemin donné et les sauvegarde dans un autre dossier.

- `to_one_hot(label, num_classes)` : Convertit une étiquette numérique en encodage one-hot.

- `load_pic(base_path)` : Charge des images à partir d'un chemin de base donné.

- `organize_images(base_path, num_classes)` : Organise les images selon leurs classes et renvoie des tenseurs et leurs étiquettes.

- `split_train_test(base_path, num_classes, test_size=0.3)` : Sépare les images en ensembles de formation et de test et renvoie des tenseurs et leurs étiquettes.

- `load_images_and_labels(csv_path, images_dir, class_to_num)` : Charge les images et leurs labels depuis un chemin donné.

## cnn.py : Description et utilisation

Le fichier `cnn.py` contient une implémentation d'un réseau de neurones convolutif (CNN) destiné à la classification d'images.

### Structure du modèle CNN

Le modèle est structuré comme suit :

- **Convolution 1**: Prend 3 canaux d'entrée (pour les images RGB), produit 16 filtres, avec un noyau de 3x3 et un padding de 1. Il est suivi d'une BatchNorm et d'une activation ReLU.
- **Convolution 2**: Prend 16 canaux d'entrée, produit 32 filtres, avec un noyau de 3x3 et un padding de 1. Il est également suivi d'une BatchNorm et d'une activation ReLU.
- **Convolution 3**: Prend 32 canaux d'entrée, produit 64 filtres, avec un noyau de 3x3 et un padding de 1. Il est de nouveau suivi d'une BatchNorm et d'une activation ReLU.
- **Couche entièrement connectée 1**: Transforme une entrée de 64 * 12 * 12 neurones en une sortie de 512 neurones.
- **Couche entièrement connectée 2**: Transforme une entrée de 512 neurones en une sortie de 256 neurones.
- **Couche entièrement connectée 3 (sortie)**: Transforme une entrée de 256 neurones en une sortie définie par `num_classes`.

### Fonctions principales

- **set_seeds(seed_value=42)**: Fixe les graines aléatoires pour assurer la reproductibilité.
- **train_model()**: Gère le processus d'entraînement du modèle. Accepte les chargeurs de données, définit l'optimiseur et le critère de perte, effectue l'entraînement et sauvegarde le modèle.
- **_save_training_graphs()**: Une fonction interne qui sauvegarde des graphiques de la progression de l'entraînement.

### Utilisation

1. **Importation** : Pour utiliser la classe `SimpleCNN` dans votre script ou notebook, importez-la depuis `cnn.py` :
    ```python
    from cnn import SimpleCNN
    ```

2. **Création d'une instance** : Initialisez le modèle en précisant le nombre de classes souhaité :
    ```python
    model = SimpleCNN(num_classes=33)
    ```

3. **Entraînement** : Après avoir préparé vos chargeurs de données, entraînez le modèle :
    ```python
    model.train_model("nom_du_modele", train_loader, epochs=20, learning_rate=0.001, val_loader=val_loader)
    ```

4. **Graphiques d'entraînement** : Après l'entraînement, les graphiques montrant la perte et la précision seront sauvegardés dans le dossier "Model/nom_du_modele/".

## Dépendances

- pandas
- PIL (Pillow)
- os
- torch
- torchvision
- tqdm
- matplotlib

## Conclusion

La bibliothèque `Lib` est un outil essentiel pour ceux qui cherchent à préparer rapidement et efficacement leurs images pour des applications de deep learning avec PyTorch. Elle fournit un ensemble de fonctions et de classes pour simplifier ce processus, de la manipulation des images à leur transformation en tenseurs utilisables avec PyTorch.

