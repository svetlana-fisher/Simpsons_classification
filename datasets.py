from imports import *

train_dataset_path = "C:\\Users\\sveta\\Documents\\Simpsons_classification\\data\\simpsons_dataset"
test_dataset_path = "C:\\Users\\sveta\\Documents\\Simpsons_classification\\data\\kaggle_simpson_testset"


class SimpsonsDataset(Dataset):
    def __init__(self, transform, dataset_path):
        self.dataset_path = dataset_path
        self.transform = transform
        self.labels_dict = dict()
        self.data = self.train_data_preparing()

    def train_data_preparing(self):
        images_labels = []
        # Сбор всех изображений и их меток
        for label, folder in enumerate(os.listdir(self.dataset_path)):
            self.labels_dict[folder] = str(label)
            folder_path = os.path.join(self.dataset_path, folder)
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                images_labels.append((img_path, label))
        return images_labels


    def __getitem__(self, item):
        img_path, label = self.data[item]
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            return None, label

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Failed to load image: {img_path}")
            return None, label

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


def get_val_subset(dataset, percent):
    class_images = {}
    for img_path, label in dataset.data:
        if label not in class_images:
            class_images[label] = []
        class_images[label].append(img_path)

    val_images = []
    for label, images in class_images.items():
        num_images = int(len(images)*percent)
        selected_images = random.sample(images, num_images)
        val_images.extend([(img, label) for img in selected_images])
    random.shuffle(val_images)
    return val_images


class ValSimpsonsDataset(Dataset):
    def __init__(self, transform, data):
        self.transform = transform
        self.data = data


    def __getitem__(self, item):
        img_path, label = self.data[item]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)



class TestSimpsonsDataset(Dataset):
    def __init__(self, transform, dataset_path):
        self.dataset_path = dataset_path
        self.transform = transform
        self.data = self.load_images()

    def load_images(self):
        images = []
        for img_name in os.listdir(self.dataset_path):
            img_path = os.path.join(self.dataset_path, img_name)
            images.append(img_path)
        return images

    def __getitem__(self, item):
        img_path = self.data[item]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, img_path

    def __len__(self):
        return len(self.data)


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.4623, 0.4075, 0.3525], [0.2562, 0.2340, 0.2646]
    )
])

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-30, 30)),
    transforms.Normalize(
        [0.4623, 0.4075, 0.3525], [0.2562, 0.2340, 0.2646]
    )
])


train_dataset = SimpsonsDataset(train_transform, train_dataset_path)
train_data = train_dataset.data
# print(len(train_dataset.data))

val_data_prep = get_val_subset(train_dataset, 0.2)
val_dataset = ValSimpsonsDataset(transform, val_data_prep)
val_data = val_dataset.data
# print(len(val_data))

train_dataset_filtered = [(img, label) for img, label in train_data if (img, label) not in val_data]
train_dataset.data = train_dataset_filtered


train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)


test_dataset = TestSimpsonsDataset(transform, test_dataset_path)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)