# Import necessary modules from PyTorch
import torchvision
import torch
from torchvision import transforms, datasets

# Import necessary modules for Neural Network
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define custom Convolution Neural Network
class CNN(nn.Module):
    def __init__(self, num_channels=3, num_out_ch=[4, 8, 16, 32], dropout=0.2, num_neurons=1024, num_classes=102):
        super(CNN, self).__init__()

        # Convolutional layers
        self.layer1 = nn.Sequential( #This is technically not a type of layer but it helps in combining different operations that are part of the same step
            nn.Conv2d(in_channels=num_channels, out_channels=num_out_ch[0], kernel_size=3, stride=1, padding=1), # Applies a 2D convolution over an input image composed of several input planes
            nn.BatchNorm2d(num_out_ch[0]), # This applies batch normalization to the output from the convolutional layer
            nn.ReLU() # Activation function is used to introduce nonlinearity in a neural network, helping mitigate the vanishing gradient problem during machine learning model training and enabling neural networks to learn more complex relationships in data
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=num_out_ch[0], out_channels=num_out_ch[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_out_ch[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Max pooling layer: down-sample an image by applying max filer to subregion
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=num_out_ch[0], out_channels=num_out_ch[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_out_ch[1]),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=num_out_ch[1], out_channels=num_out_ch[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_out_ch[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=num_out_ch[1], out_channels=num_out_ch[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_out_ch[2]),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=num_out_ch[2], out_channels=num_out_ch[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_out_ch[2]),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=num_out_ch[2], out_channels=num_out_ch[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_out_ch[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=num_out_ch[2], out_channels=num_out_ch[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_out_ch[3]),
            nn.ReLU()
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(in_channels=num_out_ch[3], out_channels=num_out_ch[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_out_ch[3]),
            nn.ReLU()
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(in_channels=num_out_ch[3], out_channels=num_out_ch[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_out_ch[3]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(in_channels=num_out_ch[3], out_channels=num_out_ch[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_out_ch[3]),
            nn.ReLU()
        )
        self.layer12 = nn.Sequential(
            nn.Conv2d(in_channels=num_out_ch[3], out_channels=num_out_ch[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_out_ch[3]),
            nn.ReLU()
        )
        self.layer13 = nn.Sequential(
            nn.Conv2d(in_channels=num_out_ch[3], out_channels=num_out_ch[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_out_ch[3]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout), # Dropout layer to prevent overfitting
            nn.Linear(7*7*num_out_ch[3], num_neurons), # Performs a matrix multiplication of the input data with the weight matrix and adding the bias term
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_neurons, num_neurons),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(num_neurons, num_classes)
        )

    # Defines the forward pass of the network, where input data x is passed through each layer sequentially.
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    

def setup_data_transforms():
    common_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),  # Randomly crop and resize the image
            transforms.RandomHorizontalFlip(),   # Randomly flip the image horizontally
            transforms.RandomRotation(10),       # Randomly rotate the image by up to 10 degrees
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),  # Randomly adjust brightness, contrast, saturation
            transforms.ToTensor(),               # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),              # Resize the image to 256x256
            transforms.CenterCrop(224),          # Crop the center of the image to 224x224
            transforms.ToTensor(),               # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])
    }
    return common_transforms

# Apply transformations to the dataset
def load_datasets(data_dir, transforms):
    datasets_flowers = {
        'train': datasets.Flowers102(root=data_dir, split='train', transform=transforms['train'], download=True),
        'val': datasets.Flowers102(root=data_dir, split='val', transform=transforms['val'], download=True),
        'test': datasets.Flowers102(root=data_dir, split='test', transform=transforms['val'], download=True)
    }
    return datasets_flowers

# Create data loaders
def create_dataloaders(datasets, batch_size, num_workers):
    loaders = {
        'train': torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        'val': torch.utils.data.DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        'test': torch.utils.data.DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    }
    return loaders



# Load and save checkpoint
def load_checkpoint(filepath, model, optimizer, scheduler):
    """ Load checkpoint if it exists, returns epoch and loss history """
    if torch.cuda.is_available():
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    epoch = checkpoint['epoch']
    train_loss_history = checkpoint.get('train_loss', [])
    val_loss_history = checkpoint.get('val_loss', [])
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    return epoch, train_loss_history, val_loss_history, best_val_loss

def save_checkpoint(filepath, model, optimizer, scheduler, epoch, train_loss_history, val_loss_history, best_val_loss):
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'best_val_loss': best_val_loss
    }, filepath)



# Define training, validation and testing functions
def train_one_epoch(model, device, dataloader, optimizer, loss_fn):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total

def validate(model, device, dataloader, loss_fn):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total

def run_training(model, device, epochs, train_loader, valid_loader, optimizer, scheduler, loss_fn, start_epoch=0, best_val_loss=float('inf'), checkpoint_path='checkpoint.pth'):
    train_loss_history, val_loss_history = [], []
    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, loss_fn)
        val_loss, val_acc = validate(model, device, valid_loader, loss_fn)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Append loss history for plotting or monitoring
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # Check if validation loss has improved and save the best model
        if val_loss < best_val_loss:
            print("Creating new checkpoint for best model...")
            best_val_loss = val_loss
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, train_loss_history, val_loss_history, best_val_loss)
        
        # Step the scheduler
        scheduler.step(val_loss)

    # Save the model at the end of training
    save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, train_loss_history, val_loss_history, best_val_loss)
    print("Train finished and the model is saved")


def test_model(model, device, test_loader, loss_fn):
    model.eval()  # Set the model to evaluation mode
    running_test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # No gradients needed for testing, reduces memory usage
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_loss = running_test_loss / total
    test_accuracy = 100.0 * correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')



def main():
    # Define hyperparameters
    NUM_EPOCHS = 1000
    BATCH_SIZE = 16
    NUM_WORKERS = 1
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    STEP_SIZE = 2
    FACTOR = 0.5

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = CNN().to(device)

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=FACTOR, patience=STEP_SIZE)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # Setup data transforms and loaders
    transforms = setup_data_transforms()
    datasets = load_datasets('dataset_flower102/', transforms)
    loaders = create_dataloaders(datasets, BATCH_SIZE, NUM_WORKERS)

    # Check for existing checkpoint and load it if available
    checkpoint_path = 'viking_finish_trained_model_TA60_VA49_T44.pth'
    try:
        start_epoch, train_loss_history, val_loss_history, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        print(f'Resuming from epoch {start_epoch + 1}')
        print("Best Val Loss: ", best_val_loss)
    except FileNotFoundError:
        start_epoch, train_loss_history, val_loss_history, best_val_loss = 0, [], [], float('inf')
    

    # Train the model (comment this code if you just want to test the model)
    # run_training(model, device, NUM_EPOCHS, loaders['train'], loaders['val'], optimizer, scheduler, loss_fn, start_epoch, best_val_loss, checkpoint_path)

    # Test the model
    test_model(model, device, loaders['test'], loss_fn)


if __name__ == "__main__":
    main()