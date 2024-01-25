import numpy as np
import cv2
import os
from image_processing import func  # Assuming func is defined in image_processing module

def resize_image(image, target_size):
    return cv2.resize(image, target_size)

if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/train"):
    os.makedirs("data/train")
if not os.path.exists("data/test"):
    os.makedirs("data/test")

path = "data/train"
path1 = "data2"
a = ['label']

for i in range(64 * 64):
    a.append("pixel" + str(i))

label = 0
var = 0
c1 = 0
c2 = 0
print("Entering the loop...")
for (dirpath, dirnames, filenames) in os.walk(path):
    print(f"Current directory path: {dirpath}")
    print(f"Subdirectories: {dirnames}")
    print(f"Files: {filenames}")
    for dirname in dirnames:
        print(f"Processing directory: {dirname}")
        for (direcpath, direcnames, files) in os.walk(os.path.join(path, dirname)):
            if not os.path.exists(os.path.join(path1, "train", dirname)):
                os.makedirs(os.path.join(path1, "train", dirname))
            if not os.path.exists(os.path.join(path1, "test", dirname)):
                os.makedirs(os.path.join(path1, "test", dirname))
            num = int(0.75 * len(files))
            print(f"Number of files in {dirname}: {len(files)}")
            print(f"Selected number of files for training: {num}")
            i = 0
            for file in files:
                var += 1
                actual_path = os.path.join(path, dirname, file)
                actual_path1 = os.path.join(path1, "train", dirname, file)
                actual_path2 = os.path.join(path1, "test", dirname, file)
                img = cv2.imread(actual_path, 0)
                bw_image = func(actual_path) 
                
                bw_image_resized = resize_image(bw_image, (310, 310))
 # Assuming func is a valid image processing function
                if i < num:
                    c1 += 1
                    cv2.imwrite(actual_path1, bw_image_resized)
                    # print(actual_path1)
                else:
                    c2 += 1
                    cv2.imwrite(actual_path2, bw_image_resized)

                i = i + 1

        label = label + 1

print("Processing complete.")
print(f"Total directories processed: {label}")
print(f"Total images processed: {var}")
print(f"Images in training set: {c1}")
print(f"Images in testing set: {c2}")
