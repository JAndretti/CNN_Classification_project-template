# ImageProcessor Class

## Présentation

La classe `ImageProcessor` est conçue pour faciliter la gestion, la transformation et l'augmentation d'images contenues dans un dossier spécifique. Elle offre des fonctionnalités essentielles pour le prétraitement d'images en vue de leur utilisation dans le contexte du machine learning, en particulier avec PyTorch.

## Fonctionnalités

- **Suppression de .DS_Store** : Retire les fichiers indésirables générés sur macOS.
- **Conversion en Tenseurs** : Transforme toutes les images du dossier en une liste de tenseurs PyTorch.
- **Normalisation** : Ajuste les tenseurs pour qu'ils aient des valeurs comprises entre 0 et 255.
- **Redimensionnement** : Change la taille de toutes les images pour qu'elles correspondent à une dimension donnée.
- **Augmentation** : Applique un ensemble de transformations pour augmenter la diversité du dataset.
- **Nuances de Gris** : Convertit les images RGB en nuances de gris.

## Utilisation

### Initialisation

Pour créer une instance de la classe:

```python
processor = ImageProcessor('chemin_vers_les_images')
```

### Traitement

Suppression des fichiers .DS_Store:

```python
processor.remove_ds_store()
```

Conversion des images en tenseurs:

```python
processor.images_to_tensor()
```

Redimensionnement des images:

```python
processor.resize_tensors(128, 128)
```

Augmentation des images:

```python
processor.augment_tensors()
```

Conversion en nuances de gris:

```python
processor.convert_to_grayscale()
```

### Accès aux Tenseurs

Pour accéder à la liste des tenseurs:

```python
tensors = processor.tensors
```

## Dépendances

- pandas
- PIL (Pillow)
- os
- torch
- torchvision
- tqdm

## Conclusion

`ImageProcessor` est un outil essentiel pour ceux qui cherchent à préparer rapidement et efficacement leurs images pour des applications de deep learning avec PyTorch.
