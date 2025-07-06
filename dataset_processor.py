import os
import shutil
import xml.etree.ElementTree as ET
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from torchvision import transforms
from torch.utils.data import Dataset


class OCRDatasetProcessor:
    def __init__(self, blank_char='-'):
        self.blank_char = blank_char
        self.chars = ""
        self.vocab_size = 0
        self.char_to_idx = {}
        self.idx_to_char = {}

    def extract_data_from_xml(self, path):
        tree = ET.parse(path)
        root = tree.getroot()

        image_paths = []
        image_sizes = []
        image_labels = []
        bounding_boxes = []

        for image in root:
            bbs_of_image = []
            labels_of_image = []

            for bbs in image.findall("taggedRectangles"):
                for bb in bbs:
                    if not bb[0].text.isalnum():
                        continue
                    if "é" in bb[0].text.lower() or "ñ" in bb[0].text.lower():
                        continue

                    bbs_of_image.append(
                        [
                            float(bb.attrib["x"]),
                            float(bb.attrib["y"]),
                            float(bb.attrib["width"]),
                            float(bb.attrib["height"]),
                        ]
                    )
                    labels_of_image.append(bb[0].text.lower())

            image_paths.append(image[0].text)
            image_sizes.append(
                (int(image[1].attrib["x"]), int(image[1].attrib["y"])))
            bounding_boxes.append(bbs_of_image)
            image_labels.append(labels_of_image)

        return image_paths, image_sizes, image_labels, bounding_boxes

    def convert_to_yolo_format(self, image_paths, image_sizes, bounding_boxes):
        yolo_data = []

        for image_path, image_size, bboxes in zip(image_paths, image_sizes, bounding_boxes):
            image_width, image_height = image_size
            yolo_labels = []

            for bbox in bboxes:
                x, y, w, h = bbox
                center_x = (x + w / 2) / image_width
                center_y = (y + h / 2) / image_height
                normalized_width = w / image_width
                normalized_height = h / image_height
                class_id = 0
                yolo_label = f"{class_id} {center_x} {center_y} {normalized_width} {normalized_height}"
                yolo_labels.append(yolo_label)

            yolo_data.append((image_path, yolo_labels))

        return yolo_data

    def save_data(self, data, src_img_dir, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

        for image_path, yolo_labels in data:
            shutil.copy(
                os.path.join(src_img_dir, image_path),
                os.path.join(save_dir, "images")
            )

            image_name = os.path.splitext(os.path.basename(image_path))[0]
            with open(os.path.join(save_dir, "labels", f"{image_name}.txt"), "w") as f:
                for label in yolo_labels:
                    f.write(f"{label}\n")

    def plot_image_with_bbs(self, img_path, bbs, labels):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for idx, bb in enumerate(bbs):
            start_point = (int(bb[0]), int(bb[1]))
            end_point = (int(bb[0] + bb[2]), int(bb[1] + bb[3]))
            color = (255, 0, 0)
            img = cv2.rectangle(img, start_point, end_point, color, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            text = labels[idx]
            text_size, _ = cv2.getTextSize(text, font, 0.5, 2)
            cv2.rectangle(
                img,
                (start_point[0], start_point[1] - text_size[1] - 10),
                (start_point[0] + text_size[0], start_point[1]),
                color, cv2.FILLED
            )
            cv2.putText(
                img, text, (start_point[0], start_point[1] - 5),
                font, 0.5, (255, 255, 255), 2
            )
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    def split_bounding_boxes(self, img_paths, img_labels, bboxes, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        count = 0
        labels = []

        for img_path, img_label, bbs in zip(img_paths, img_labels, bboxes):
            img = Image.open(img_path)

            for label, bb in zip(img_label, bbs):
                cropped_img = img.crop(
                    (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))
                if np.mean(cropped_img) < 35 or np.mean(cropped_img) > 220:
                    continue
                if cropped_img.size[0] < 10 or cropped_img.size[1] < 10:
                    continue

                filename = f"{count:06d}.jpg"
                cropped_img.save(os.path.join(save_dir, filename))

                label_line = f"{os.path.join(save_dir, filename)}\t{label}"
                labels.append(label_line)
                count += 1

        print(f"Created {count} cropped word images")
        with open(os.path.join(save_dir, "labels.txt"), "w") as f:
            for line in labels:
                f.write(f"{line}\n")

    def load_image_paths_and_labels(self, root_dir):
        img_paths, labels = [], []
        with open(os.path.join(root_dir, "labels.txt"), "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    img_paths.append(parts[0])
                    labels.append(parts[1])
        print(f"Total word images: {len(img_paths)}")
        return img_paths, labels

    def build_vocab_from_labels(self, labels):
        letters = [char.split(".")[0].lower() for char in labels]
        letters = "".join(letters)
        unique_letters = sorted(set(letters))
        chars = "".join(unique_letters) + self.blank_char
        self.chars = chars
        self.vocab_size = len(chars)
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(chars)}
        print(f"Vocab: {chars}\nVocab size: {self.vocab_size}")
        return chars, self.vocab_size

    def encode(self, label, max_label_len):
        encoded = torch.tensor(
            [self.char_to_idx[char] for char in label], dtype=torch.long
        )
        label_len = torch.tensor(len(encoded), dtype=torch.long)
        padded = F.pad(encoded, (0, max_label_len - len(encoded)), value=0)
        return padded, label_len

    def decode(self, encoded_sequences):
        decoded = []
        for seq in encoded_sequences:
            label = []
            prev_char = None
            for token in seq:
                if token != 0:
                    char = self.idx_to_char[token.item()]
                    if char != self.blank_char:
                        if char != prev_char or prev_char == self.blank_char:
                            label.append(char)
                    prev_char = char
            decoded.append("".join(label))
        return decoded

    def get_transforms(self):
        return {
            "train": transforms.Compose([
                transforms.Resize((100, 420)),
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.Grayscale(1),
                transforms.GaussianBlur(3),
                transforms.RandomAffine(degrees=1, shear=1),
                transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                transforms.RandomRotation(degrees=2),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]),
            "val": transforms.Compose([
                transforms.Resize((100, 420)),
                transforms.Grayscale(1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]),
        }


class STRDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        char_to_idx,
        max_label_len,
        label_encoder=None,
        transform=None,
    ):
        self.transform = transform
        self.img_paths = X
        self.labels = y
        self.char_to_idx = char_to_idx
        self.max_label_len = max_label_len
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.label_encoder:
            encoded_label, label_len = self.label_encoder(
                label, self.char_to_idx, self.max_label_len
            )
        return img, encoded_label, label_len
