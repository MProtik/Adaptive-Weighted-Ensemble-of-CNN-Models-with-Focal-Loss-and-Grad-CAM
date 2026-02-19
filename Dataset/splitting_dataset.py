import os
import shutil
import random
from tqdm import tqdm

src_dir = "/content/drive/MyDrive/Research paper on cnn and ensemble learning/Dataset/Folder according to classes"
output_dir = "/content/drive/MyDrive/Research paper on cnn and ensemble learning/Dataset/Splitted_dataset"

classes = []

for cls in os.listdir(src_dir):
  classes.append(cls)

random.seed(42)

for cls in classes:
  class_dir = os.path.join(src_dir, cls)
  images = os.listdir(class_dir)
  random.shuffle(images)

  total_number = len(images)
  train_number = int(total_number * 0.8)
  val_number = int(total_number * 0.1)

  train_images = images[:train_number]
  val_images =  images[train_number: train_number + val_number]
  test_images = images[train_number + val_number:]

  for split_name, split_images in zip(["train", "val", "test"], [train_images, val_images, test_images]):
    split_dir = os.path.join(output_dir, split_name, cls)
    os.makedirs(split_dir)
    for img in tqdm(split_images):
      src = os.path.join(class_dir, img)
      dst = os.path.join(split_dir, img)
      shutil.copy2(src, dst)

print("\n✅ Dataset successfully split into train/val/test folders!")
print(f"Output directory: {os.path.abspath(output_dir)}")

print(f"Output directory: {os.path.abspath(output_dir)}")