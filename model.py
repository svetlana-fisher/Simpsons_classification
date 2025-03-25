from imports import *
from datasets import val_loader


class SimpsonsNet(nn.Module):
    def __init__(self, num_classes=42):
        super(SimpsonsNet, self).__init__()
        self.model = torchvision.models.resnet18(weights='DEFAULT').cuda()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes).cuda()

    def forward(self, img):
        return self.model(img)


writer = SummaryWriter()
def train(model, train_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float('inf')
    patience = 6
    wait = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, lables in train_loader:
            images, lables = images.cuda(), lables.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, lables)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        # Логирование в TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Val Acc: {val_acc:.6f}")

        if avg_val_loss < best_val_loss:
            wait = 0
            best_val_loss = avg_val_loss
            # torch.save(model.state_dict(), "best.pth")
            # torch.save(model.state_dict(), "best_normalize.pth")
            torch.save(model.state_dict(), "best_flip_turn.pth")
        else:
            wait += 1
            if wait >= patience:
                print("model stopped learning. early stop")
                break

    writer.close()