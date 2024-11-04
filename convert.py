import os
import json
import cv2

def convert_json_to_yolo(json_file, image_width, image_height, output_dir):
    """
    Converts a JSON label file to YOLO format and saves it to the output directory.

    Args:
        json_file: Path to the JSON file.
        image_width: Width of the image.
        image_height: Height of the image.
        output_dir: Output directory for the converted .txt files.
    """

    with open(json_file, 'r') as f:
        data = json.load(f)

    for obj in data:
        x = obj['x']
        y = obj['y']
        width = obj['width']
        height = obj['height']

        # Calculate center coordinates and normalize
        x_center = (x + width / 2) / image_width
        y_center = (y + height / 2) / image_height
        width = width / image_width
        height = height / image_height

    # Create the output filename based on the input filename
    output_file = os.path.splitext(os.path.basename(json_file))[0] + '.txt'
    output_path = os.path.join(output_dir, output_file)

    # Adjust class ID to start from 0
    # class_id = int(obj['class_id']) - 1 
    class_id = 0

    with open(output_path, 'w') as f:
        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        print(f"Wrote label: {class_id} {x_center} {y_center} {width} {height} to {output_path}")


# Specify the root directory of your dataset
dataset_root = 'theos-guns'

# Iterate through train, validation, and test sets
for dataset_type in ['train', 'valid', 'test']:
    image_dir = os.path.join(dataset_root, dataset_type, 'images')
    label_dir = os.path.join(dataset_root, dataset_type, 'labels')
    output_dir = os.path.join(dataset_root, dataset_type, 'labels')  # save in same directory

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(label_dir):
        if filename.endswith('.json'):
            json_file = os.path.join(label_dir, filename)
            image_file = os.path.join(image_dir, os.path.splitext(filename)[0] + '.jpg')  # Assuming .jpg images

            # Get image dimensions (replace with your method if needed)
            # You can use libraries like OpenCV or PIL to get image dimensions
            # For example:
            # import cv2
            img = cv2.imread(image_file)
            image_width, image_height = img.shape[:2]
            
            print(f"Image dimensions: {image_width}x{image_height}")

            # Replace with your actual image width and height
            # image_width = 640
            # image_height = 480

            convert_json_to_yolo(json_file, image_width, image_height, output_dir)