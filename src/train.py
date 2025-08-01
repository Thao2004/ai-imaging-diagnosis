import torch
from torch.utils.data import DataLoader
from src.model import get_resnet_model
from src.dataset import MedicalImageDataset
from src.transforms import get_transforms
from src.utils import save_model

def train_model(train_data, val_data, num_classes, epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet_model(num_classes).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    save_model(model, "resnet_medical.pt")
