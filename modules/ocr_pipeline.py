import torch
import matplotlib.pyplot as plt
from PIL import Image


class OCRPipeline:
    def __init__(
        self,
        text_det_model,
        text_reg_model,
        data_transforms,
        idx_to_char,
        blank_char="-",
        device="cuda",
    ):
        self.text_det_model = text_det_model
        self.text_reg_model = text_reg_model
        self.data_transforms = data_transforms
        self.idx_to_char = idx_to_char
        self.blank_char = blank_char
        self.device = device

    def decode(self, encoded_sequences):
        decoded_sequences = []
        for seq in encoded_sequences:
            decoded_label = []
            prev_char = None
            for token in seq:
                if token != 0:
                    char = self.idx_to_char[token.item()]
                    if char != self.blank_char:
                        if char != prev_char or prev_char == self.blank_char:
                            decoded_label.append(char)
                    prev_char = char
            decoded_sequences.append("".join(decoded_label))
        return decoded_sequences

    def text_detection(self, img_path):
        results = self.text_det_model(img_path, verbose=False)[0]
        bboxes = results.boxes.xyxy.tolist()
        classes = results.boxes.cls.tolist()
        names = results.names
        confs = results.boxes.conf.tolist()
        return bboxes, classes, names, confs

    def text_recognition(self, img):
        transformed_image = self.data_transforms(img)
        transformed_image = transformed_image.unsqueeze(0).to(self.device)

        self.text_reg_model.eval()
        with torch.no_grad():
            logits = self.text_reg_model(transformed_image).detach().cpu()

        decoded = self.decode(logits.permute(1, 0, 2).argmax(2))
        return decoded[0]

    def visualize_detections(self, img, detections):
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis("off")

        for bbox, detected_class, confidence, transcribed_text in detections:
            x1, y1, x2, y2 = bbox
            plt.gca().add_patch(
                plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2
                )
            )
            plt.text(
                x1,
                y1 - 10,
                f"{detected_class} ({confidence:.2f}): {transcribed_text}",
                fontsize=9,
                bbox=dict(facecolor="red", alpha=0.5),
            )

        plt.show()

    def predict(self, img_path):
        # Detection
        bboxes, classes, names, confs = self.text_detection(img_path)
        img = Image.open(img_path)

        predictions = []

        # Iterate boxes -> crop -> recog
        for bbox, cls, conf in zip(bboxes, classes, confs):
            x1, y1, x2, y2 = bbox
            detected_class = names[int(cls)]
            cropped_image = img.crop((x1, y1, x2, y2))

            transcribed_text = self.text_recognition(cropped_image)

            predictions.append((bbox, detected_class, conf, transcribed_text))

        # Visualize
        self.visualize_detections(img, predictions)

        return predictions
