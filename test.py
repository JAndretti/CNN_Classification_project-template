from Lib import load_images_and_labels, split_train_test, SimpleCNN,organize_images
import torch
import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


# csv_path = 'data2/sampleSubmission.csv'
images_dir = 'data2/test/test'
outpout_dir_train = 'data2/train/train'

nbr_classes = 11

data, class_to_idx, paths = organize_images(images_dir,nbr_classes)
# data = load_images_and_labels(csv_path, images_dir, class_to_idx)


# 1. Charger le modèle
model_path = 'Model/50epochs_100x100_lr0,0001/50epochs_100x100_lr0,0001.pth'
model = SimpleCNN(num_classes=nbr_classes)
model.load_state_dict(torch.load(model_path))
model.eval()  # Mettre le modèle en mode évaluation

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu")
print(f"testing on {device}")
model.to(device)

# 2. Passez les données à travers le modèle
correct = 0
total = len(data)

true_labels = []
pred_labels = []

# Comptez le nombre de représentants par classe
label_counts = defaultdict(int)

with torch.no_grad():  # Nous n'avons pas besoin de gradients pour l'évaluation
    for tensor, label in tqdm.tqdm(data):
        tensor = tensor.unsqueeze(0).to(device)  # Ajoutez une dimension batch et transférez sur le device
        output = model(tensor)
        _, predicted = torch.max(output.data, 1)
        value = torch.argmax(label.data)
        
        # Augmentez le compte pour cette classe
        class_name = [k for k, v in class_to_idx.items() if v == value.item()][0]
        label_counts[class_name] += 1
        
        # Ajout des labels pour la matrice de confusion
        true_labels.append(value.item())
        pred_labels.append(predicted.cpu().item())

        correct += (predicted.cpu() == value).sum().item()

# Imprimez les comptes
print("\nNombre de représentants par classe:")
for class_name, count in label_counts.items():
    print(f"{class_name}: {count}")
# 3. Calculez la précision
accuracy = 100 * correct / total
print(f"Accuracy on test data: {accuracy:.2f}%")

# 4. Matrice de confusion
cm = confusion_matrix(true_labels, pred_labels)

# Augmenter la taille de la figure
fig, ax = plt.subplots(figsize=(12, 10))

cax = ax.matshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matrice de Confusion')
plt.colorbar(cax)
tick_marks = np.arange(len(class_to_idx))
plt.xticks(tick_marks, list(class_to_idx.keys()), rotation=45, ha="right")
plt.yticks(tick_marks, list(class_to_idx.keys()))
plt.ylabel('Vraies étiquettes')
plt.xlabel('Étiquettes prédites')

# Espacer les étiquettes
ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_label_position('bottom')
plt.tight_layout()

# Sauvegardez la matrice de confusion
save_path = model_path.rsplit('/', 1)[0] + "/confusion_matrix.png"
plt.savefig(save_path)
print(f"Matrice de confusion saved at {save_path}")