from Lib import data_process, split_train_test, SimpleCNN
from torch.utils.data import DataLoader

num_classes = 11

outpout_dir_train = 'data2/train/train'
outpout_dir_test = 'data2/test/test'

train_data, test_data, class_to_idx, train_paths, test_paths = split_train_test(
    outpout_dir_train, num_classes)

# Create DataLoader instances
batch_size = 10
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Initialize the CNN model
model = SimpleCNN(num_classes=num_classes)

# Train the model
model.train_model('50epochs_100x100_lr0,0001', train_loader,
                  learning_rate=0.0001, epochs=50, val_loader=test_loader)

print("Training complete!")
