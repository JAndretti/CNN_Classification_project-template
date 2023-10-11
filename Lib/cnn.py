import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import time


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=33):
        super(SimpleCNN, self).__init__()

        # Couche de convolution 1 : 3 canaux d'entrée pour les images RGB, 16 filtres/canaux de sortie, taille du noyau 3x3
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Couche de convolution 2 : 16 canaux d'entrée, 32 filtres/canaux de sortie, taille du noyau 3x3
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Couche de convolution 3 : 32 canaux d'entrée, 64 filtres/canaux de sortie, taille du noyau 3x3
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Couche entièrement connectée (ou Dense layer)
        # Après 3 couches de convolution et 3 couches de max-pooling de taille 2x2, la taille de l'image est réduite à [64, 6, 6]
        # Couche entièrement connectée
        self.fc1 = nn.Linear(64 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 256)  # nouvelle taille de sortie
        self.fc3 = nn.Linear(256, num_classes)  # nouvelle couche

    def forward(self, x):
        # Couche 1 avec activation ReLU et max-pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        # Couche 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        # Couche 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        # Mise à plat des données avant la couche entièrement connectée
        x = x.view(x.size(0), -1)

        # Couche entièrement connectée
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # nouvelle étape
        # # Appliquer softmax sur la dimension des classes
        # x = F.softmax(x, dim=1)

        return x

    @staticmethod
    def set_seeds(seed_value=42):
        """Fixe les graines aléatoires pour la reproductibilité."""
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)

    def train_model(self, name, train_loader, epochs=10, learning_rate=0.001, val_loader=None, save_path="Model/"):
        self.set_seeds()
        self.name = name
        save_path += self.name
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Déterminer si le GPU est disponible
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Training on {device}")

        # Déplacer le modèle vers le GPU si disponible
        self.to(device)

        # Définir le critère de perte et l'optimiseur
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        best_accuracy = 0  # Pour suivre la meilleure précision et enregistrer le meilleur modèle
        loss_history = []
        accuracy_history = []
        # Boucle d'entraînement
        for epoch in range(epochs):
            start = time.time()
            running_loss = 0.0
            self.train()  # Mettre le modèle en mode d'entraînement
            # Pour stocker les pertes moyennes et les précisions pour la création de graphiques

            for i, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Calculer la perte moyenne pour cette époque
            average_loss = running_loss / len(train_loader)

            # Afficher la progression
            print(
                f"Epoch [{epoch + 1}/{epochs}] - Loss: {average_loss:.4f} en {time.ctime(time.time() - start)[11:19]}", end="")

        # Sauvegarde des métriques pour la création de graphiques
            loss_history.append(average_loss)
            if val_loader:
                correct = 0
                total = 0
                self.eval()  # Mettre le modèle en mode d'évaluation
                with torch.no_grad():
                    for data in val_loader:
                        images, labels = data[0].to(device), data[1].to(device)
                        outputs = self(images)
                        _, predicted = torch.max(outputs.data, 1)
                        _, value = torch.max(labels.data, 1)
                        total += labels.size(0)
                        correct += (predicted == value).sum().item()
                accuracy = 100 * correct / total
                accuracy_history.append(accuracy)
                print(f" - Validation Accuracy: {accuracy:.2f}%")

                # Enregistre le modèle si c'est la meilleure précision jusqu'à présent
                if accuracy > best_accuracy and epoch % 100 == 0:
                    best_accuracy = accuracy
                    torch.save(self.state_dict(), save_path +
                               '/'+self.name + '.pth')
                    print(
                        f" > Model saved at {save_path+'/'+self.name+'.pth'}")

            else:
                print("")
        torch.save(self.state_dict(), save_path + '/'+self.name + '.pth')
        print(f" > Model saved at {save_path+'/'+self.name+'.pth'}")
        # Enregistrement des graphiques après l'entraînement
        self._save_training_graphs(epochs, loss_history, accuracy_history)

        print("Class CNN DONE")

    def _save_training_graphs(self, epochs, loss_history, accuracy_history, save_path='Model/'):
        save_path += self.name+'/'
        # Création du dossier s'il n'existe pas
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        x = [val for val in range(epochs)]

        # Graphique de la perte
        plt.figure()
        plt.plot(x, loss_history, label='Training Loss')
        plt.title('Training Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(save_path, 'loss_graph.png'))

        # Graphique de la précision si elle est fournie
        if accuracy_history:
            plt.figure()
            plt.plot(x, accuracy_history, label='Validation Accuracy')
            plt.title('Validation Accuracy per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.savefig(os.path.join(save_path, 'accuracy_graph.png'))
