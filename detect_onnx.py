import sys
import onnxruntime as ort
import numpy as np
import cv2
import os

# Load ONNX model
session = ort.InferenceSession("runs/train/exp13/weights/best.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Preprocess image
def preprocess(img_path, img_size=640):
    img = cv2.imread(img_path)
    print(img.flatten())
    if img is None:
        raise ValueError(f"Error loading image: {img_path}")
    
    original_height, original_width = img.shape[:2]
    im0 = img.copy()  # Keep original image
    img = cv2.resize(img, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img, im0, original_width, original_height

# Non-Max Suppression (NMS)
def nms(predictions, conf_threshold=0.25, iou_threshold=0.45):
    boxes, scores, class_ids = [], [], []

    for pred in predictions[0]:  # YOLOv5 ONNX output (1,25200,6)
        x, y, w, h, conf, cls = pred
        if conf > conf_threshold:
            boxes.append([x, y, w, h])  # Keep YOLO format for now
            scores.append(float(conf))
            class_ids.append(int(cls))

    if len(boxes) == 0:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
    if indices is None or len(indices) == 0:
        return []

    return [(boxes[i], scores[i], class_ids[i]) for i in indices.flatten()]

# Run inference and save image
def detect(img_path):
    img, im0, orig_w, orig_h = preprocess(img_path)
    print(img.flatten())
    np.savetxt("input.txt", img.flatten(), fmt='%4.6f', delimiter=',')
    # Run ONNX inference
    outputs = session.run([output_name], {input_name: img})[0]

    # Apply NMS
    detections = nms(outputs)

    if len(detections) == 0:
        print("No objects detected.")
    else:
        x_scale, y_scale = orig_w / 640, orig_h / 640  # Scale factors

        for (box, score, class_id) in detections:
            center_x, center_y, width, height = box

            # Convert YOLO format to OpenCV format (and scale back to original image size)
            x1 = int((center_x - width / 2) * x_scale)
            y1 = int((center_y - height / 2) * y_scale)
            x2 = int((center_x + width / 2) * x_scale)
            y2 = int((center_y + height / 2) * y_scale)

            label = f"{class_id}: {score:.2f}"
            cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the result with "_teaser" suffix
    base, ext = os.path.splitext(img_path)
    output_path = f"{base}_teaser{ext}"
    cv2.imwrite(output_path, im0)
    print(f"Saved output image: {output_path}")

# Main entry
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_onnx.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found!")
        sys.exit(1)

    detect(image_path)
