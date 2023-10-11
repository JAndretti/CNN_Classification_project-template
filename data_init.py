from Lib import data_process, split_train_test, SimpleCNN
import os
import shutil
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
from PIL import Image


train_dir = 'data'
# test_dir = 'data/test/test'
outpout_dir_train = 'data2/train/train'
outpout_dir_test = 'data2/test/test'

data_process(train_dir,outpout_dir_train)
# data_process(test_dir,outpout_dir_test)

train_data, test_data, class_to_idx, train_paths, test_paths = split_train_test(outpout_dir_train)


# Maintenant, déplacez les images de test vers data2/test/test organisé par classe
for path in test_paths:
    # Charger l'image sous forme de tenseur
    with Image.open(path) as img:
        tensor = ToTensor()(img)

    class_name = path.split('/')[-2]  # Récupère le nom de la classe depuis le chemin
    dest_folder = os.path.join(outpout_dir_test, class_name)
    os.makedirs(dest_folder, exist_ok=True)  # Créer le dossier de classe s'il n'existe pas
    
    dest_path = os.path.join(dest_folder, path.split('/')[-1])
    save_image(tensor, dest_path)
    
    # Supprimer l'image originale après la sauvegarde
    os.remove(path)