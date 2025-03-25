from imports import *
from datasets import test_loader, train_dataset, train_loader
from model import SimpsonsNet, train

labels_dict = train_dataset.labels_dict


def extract_label_from_filename(paths):
    lables = []
    for path in paths:
        for name in labels_dict.keys():
            if name in path:
                lables.append(labels_dict[name])
    return lables


def predict(model, test_loader):
    model.eval()
    predictions = []
    image_paths = []

    with torch.no_grad():
        for images, paths in test_loader:
            images = images.cuda()
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            predictions.extend(preds.cpu().numpy())
            image_paths.extend(paths)

    return predictions, image_paths


def show_predictions(image_paths, preds, labels_dict, num_images=8):
    fig, axes = plt.subplots(1, num_images, figsize=(20, 5))

    for i in range(num_images):
        img = Image.open(image_paths[i])
        pred_label = int(preds[i])
        real_label = None
        for img_name in labels_dict.keys():
            if img_name in image_paths[i]:
                real_label = labels_dict[img_name]

        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"True: {real_label}, Pred: {pred_label}")

    plt.show()


model = SimpsonsNet().cuda()

# train(model, train_loader, 55)
model.load_state_dict(torch.load('best_norm.pth'))
preds, paths = predict(model, test_loader)
lables = extract_label_from_filename(paths)


print(labels_dict)
print(metrics.classification_report(list(map(int, lables)), list(map(int, preds)), target_names=list(labels_dict.values())))

show_predictions(paths, preds, train_dataset.labels_dict)
