import os
import matplotlib.pyplot as plt
from datasets import train_dataset

lables_dict = train_dataset.labels_dict

data_path = "C:\\Users\\sveta\\Documents\\Simpsons_classification\\data\\simpsons_dataset"

folder_counts = {}

for folder_name in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder_name)
    folder_counts[lables_dict[folder_name]] = len(os.listdir(folder_path))

folders = list(folder_counts.keys())
counts = list(folder_counts.values())

plt.figure(figsize=(10, 6))  # Уменьшаем размер графика
bars = plt.bar(folders, counts, width=0.7)  # Увеличиваем ширину столбцов
plt.xlabel('Классы', fontsize=10)
plt.ylabel('Количество картинок', fontsize=10)
plt.title('Количество картинок в каждом классе', fontsize=12)

# Устанавливаем метки по оси X горизонтально и уменьшаем размер шрифта
plt.xticks(rotation=0, fontsize=10)  # Убираем наклон и уменьшаем размер шрифта
plt.yticks(fontsize=12)

# Устанавливаем верхнюю границу по оси Y
plt.ylim(0, max(counts) * 1.1)  # Увеличиваем границу на 10%

# Добавляем количество картинок над каждым столбцом с вертикальной ориентацией
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, (yval + 0.05 * max(counts))-80,  # Увеличиваем высоту текста
             int(yval), ha='center', va='bottom', fontsize=9, rotation=90)

plt.tight_layout()  # Чтобы избежать наложения меток
plt.show()
