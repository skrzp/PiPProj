import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import ImagesDataset
from architecture import MyCNN

def compute_accuracy(data_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _, _ in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps') # <-
    else:
        return torch.device('cpu')

def main():
    device = get_device()

    dataset = ImagesDataset(image_dir="training_data")
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_classes = len(dataset.classnames_to_ids)
    
    model = MyCNN(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels, _, _ in tqdm(data_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataset)
        
        accuracy = compute_accuracy(data_loader, model, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

    torch.save(model.state_dict(), "model.pth")

if __name__ == '__main__':
    main()