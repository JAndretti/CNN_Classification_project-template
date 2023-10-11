from Lib import data_process, split_train_test, SimpleCNN,load_pic
import torch
import tqdm

photo_dir = 'test_data'
photo_out = 'test_transform_data'

data_process(photo_dir,photo_out)

pic = load_pic(photo_out)

# 1. Charger le modèle
model_path = 'Model/10epochs_100x100/10epochs_100x100.pth'
model = SimpleCNN()
model.load_state_dict(torch.load(model_path))
model.eval()  # Mettre le modèle en mode évaluation

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu")
print(f"testing on {device}")
model.to(device)

res = []
with torch.no_grad():  # Nous n'avons pas besoin de gradients pour l'évaluation
    for tensor, label in pic:
        tensor = tensor.unsqueeze(0).to(device)  # Ajoutez une dimension batch et transférez sur le device
        output = model(tensor)
        _, predicted = torch.max(output.data, 1)
        print(predicted)        

