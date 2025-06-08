import os
import re
import cv2
import json
import shutil
import random

def json_read_file(file):
    with open(file) as f:
        d = json.load(f)
    return d

dataset_path = "CvcMuscima-Distortions/"

dest_path = "dataset"

if os.path.exists(dest_path):
    shutil.rmtree(dest_path)

os.mkdir(dest_path)
os.mkdir(os.path.join(dest_path, "images"))
os.mkdir(os.path.join(dest_path, "images", "train"))
os.mkdir(os.path.join(dest_path, "images", "val"))
os.mkdir(os.path.join(dest_path, "labels"))
os.mkdir(os.path.join(dest_path, "labels", "train"))
os.mkdir(os.path.join(dest_path, "labels", "val"))

all_jsons = os.listdir(dataset_path + "json")
random.shuffle(all_jsons)

for i, file in enumerate(all_jsons):
    is_training = i < len(all_jsons) * 0.8
    train_val_dir = "train"
    if not is_training:
        train_val_dir = "val"

    match = re.search("CVC-MUSCIMA_W-(\\d+)_N-(\\d+)_D-ideal.json$", file) 
    if not match: 
        raise ValueError("Unknown file " + file)
    dir_name = "w-" + match[1]
    img_name = "p0" + match[2] + ".png"
    img_path = os.path.join(dataset_path, "ideal", dir_name, "image", img_name)
    if not os.path.exists(img_path):
        raise ValueError("Failed to find file " + img_path)
    img = cv2.imread(img_path)
    descr = json_read_file(os.path.join(dataset_path, "json", file))
    expected_shape = (descr["width"], descr["height"])
    actual_shape = (img.shape[1], img.shape[0])
    if expected_shape != actual_shape:
        raise ValueError("Image sizes don't agree for file " + file)
    
    img_width = img.shape[1]
    img_height = img.shape[0]
    
    classes = []
    for staff in descr["staves"]:
        left = staff["left"]
        top = staff["top"]
        right = staff["right"]
        bottom = staff["bottom"]

        width = right - left
        height = bottom - top
        centerx = int(left + width / 2)
        centery = int(top + height / 2)
        if centerx > img_width or width > img_width or centery > img_height or height > img_height:
            raise ValueError("Invalid coordinates " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom) + " " + str(img_width) + " " + str(img_height) + " " + img_path)
        classes.append("0 " + " " + str(centerx / img_width) + " " + str(centery / img_height) + " " + str(width / img_width) + " " + str(height / img_height))

    with open(os.path.join(dest_path, "labels", train_val_dir, file.replace(".json", ".txt")), "w") as text_file:
        text_file.write(str.join("\n", classes))
    
    shutil.copyfile(img_path, os.path.join(dest_path, "images", train_val_dir, file.replace(".json", ".png")))