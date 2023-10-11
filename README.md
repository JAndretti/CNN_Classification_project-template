# CNN_Classification_project-template

## Structure du Projet
```
- CNN_Classification_project-template
    |-- Model               
    |-- data                
    |-- test.py             
    |-- test_photos.py      
    |-- train.py
    |-- Lib
        |-- ImageProcessor.py 
        |-- __init__.py       
        |-- cnn.py            
        |-- data_prep.py                 
    |-- data2               
    |-- data_init.py        
    |-- test_data           
    |-- test_transform_data
```

**Note**: Certains dossiers comme `Model`, `data2`, et `test_transformed_data` ne sont pas présents initialement dans le dépôt et sont générés par les scripts du projet.

## Description

- `data`: Contient des sous-dossiers avec des photos pour chaque classe. Chaque sous-dossier doit être nommé selon le nom de la classe qu'il représente.

- `Lib`: Contient les classes et fonctions essentielles pour le fonctionnement du projet.
  - `cnn.py`: Contient les classes et fonctions nécessaires pour créer et entraîner le modèle CNN.
  - `ImageProcessor.py`: Contient une classe pour transformer les images afin d'améliorer l'entraînement.
  - `data_prep.py`: Contient des fonctions pour charger les données et créer les étiquettes pour l'entraînement.

### Scripts

- `data_init.py`: 
  - Traite les images initiales et effectue les transformations nécessaires.
  - Divise les données en ensembles d'entraînement et de test.
  - Stocke le tout dans `data2`.

- `train.py`: 
  - Initialise le modèle CNN.
  - Entraîne le modèle avec les données d'entraînement.
  - Sauvegarde le modèle et les graphiques de formation dans `Model`.

- `test.py`: 
  - Charge le modèle entraîné.
  - Teste le modèle sur les données de test.
  - Crée une matrice de confusion pour évaluer les performances du modèle.
  - Sauvegarde la matrice de confusion.

- `test_photos.py`: 
  - Semblable à `test.py` mais conçu pour tester des photos individuelles ou en petits groupes.
  - Les photos à tester doivent être placées dans le dossier `test_data`.

## Comment Utiliser

1. **Préparation des Données**:
   - Placez vos photos dans le dossier `data` en organisant chaque classe dans des sous-dossiers distincts.
   - Exécutez `data_init.py` pour traiter et diviser les données.

2. **Entraînement**:
   - Lancez `train.py` pour commencer l'entraînement. Après l'entraînement, le modèle sera sauvegardé dans `Model`.

3. **Test**:
   - Utilisez `test.py` pour tester le modèle sur l'ensemble de test et obtenir une matrice de confusion.
   - Pour tester des photos individuelles ou en petits groupes, placez-les dans `test_data` et exécutez `test_photos.py`.

