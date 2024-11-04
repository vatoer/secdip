import os

def filter_missing_labels(image_dir, label_dir):
  """
  Filters images with missing label files in the training data.

  Args:
      image_dir: Path to the directory containing images.
      label_dir: Path to the directory containing label files.
  """  
  image_files = os.listdir(image_dir)
  for image_file in image_files:
    label_file = os.path.splitext(image_file)[0] + '.txt'  # Assuming .txt labels
    label_path = os.path.join(label_dir, label_file)
    if not os.path.isfile(label_path):
      image_path = os.path.join(image_dir, image_file)
      # Remove image (replace with a print statement for verification first)
      os.remove(image_path)
      print(f"Removed image: {image_file} (missing label)")

# Example usage (replace paths with your actual directories)
image_dir = './theos-guns/train/images'
label_dir = './theos-guns/train/labels'

filter_missing_labels(image_dir, label_dir)