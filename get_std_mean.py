from imports import *
from datasets import train_data


def mean_and_std():

    amount_of_pictures = len(train_data)
    r_sum = 0.0
    g_sum = 0.0
    b_sum = 0.0

    for img_path, label in train_data:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (244, 244))

        r_sum += img[:, :, 0].mean()
        g_sum += img[:, :, 1].mean()
        b_sum += img[:, :, 2].mean()

    mean_r = r_sum / amount_of_pictures
    mean_g = g_sum / amount_of_pictures
    mean_b = b_sum / amount_of_pictures

    std_r = 0.0
    std_g = 0.0
    std_b = 0.0

    for img_path, label in train_data:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (244, 244))

        std_r += np.sum((img[:, :, 0] - mean_r) ** 2)
        std_g += np.sum((img[:, :, 1] - mean_g) ** 2)
        std_b += np.sum((img[:, :, 2] - mean_b) ** 2)

    amount_of_pixels = amount_of_pictures * 244 * 244
    std_r = math.sqrt(std_r / amount_of_pixels) / 255
    std_g = math.sqrt(std_g / amount_of_pixels) / 255
    std_b = math.sqrt(std_b / amount_of_pixels) / 255

    return mean_r / 255, mean_g / 255, mean_b / 255, std_r, std_g, std_b


a = mean_and_std()
# (np.float64(0.46236819863845635), np.float64(0.40753904320958495), np.float64(0.3525630463091771), 0.2562128592147357, 0.2340181893162081, 0.26469960560235933)
print(a)