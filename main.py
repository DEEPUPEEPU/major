import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from dino_utils import load_dino_model, apply_grounded_dino
from segmentation_utils import load_sam_model, apply_sam_segmentation
from indices_utils import calculate_ndvi, calculate_vari, calculate_gli, calculate_lndvi, calculate_exg

# Directories
DATA_DIR = "../archive"
OUTPUT_DIR = "../output"

# Load models
dino_model, dino_processor = load_dino_model()
sam_model = load_sam_model()

# Function to validate bounding box
def is_valid_box(box):
    x_min, y_min, x_max, y_max = box.tolist()
    return x_min < x_max and y_min < y_max

# Process each class directory
for cls in os.listdir(DATA_DIR):
    cls_dir = os.path.join(DATA_DIR, cls)

    if not os.path.isdir(cls_dir):
        continue

    print(f"Processing class: {cls}")

    # Create corresponding output directory
    output_cls_dir = os.path.join(OUTPUT_DIR, cls)
    os.makedirs(output_cls_dir, exist_ok=True)

    for img_name in os.listdir(cls_dir)[:5]:  # Limit processing for testing
        img_path = os.path.join(cls_dir, img_name)

        try:
            # Open image and resize for efficiency
            image = Image.open(img_path).resize((512, 512))

            # Step 1: Apply Grounded DINO to detect leaves
            pred_boxes, pred_scores = apply_grounded_dino(image, dino_model, dino_processor)

            # Filter valid boxes
            valid_boxes = [box for box in pred_boxes if is_valid_box(box)]
            valid_boxes = valid_boxes[:10]  # Limit to top 10 boxes

            # Step 2: Apply SAM segmentation on the detected boxes
            segmented_image = apply_sam_segmentation(image, valid_boxes, sam_model)

            # Step 3: Calculate NDVI, LNDVI, VARI, GLI, and ExG indices
            ndvi = calculate_ndvi(np.array(segmented_image))
            lndvi = calculate_lndvi(np.array(segmented_image))
            vari = calculate_vari(np.array(segmented_image))
            gli = calculate_gli(np.array(segmented_image))
            exg = calculate_exg(np.array(segmented_image))

            # Step 4: Save the processed image
            output_path = os.path.join(output_cls_dir, img_name)
            segmented_image.save(output_path)

            # Save health indices visualization (optional)
            plt.figure(figsize=(15, 3))
            plt.subplot(1, 5, 1)
            plt.title("NDVI")
            plt.imshow(ndvi, cmap='RdYlGn')
            plt.colorbar()

            plt.subplot(1, 5, 2)
            plt.title("LNDVI")
            plt.imshow(lndvi, cmap='RdYlGn')
            plt.colorbar()

            plt.subplot(1, 5, 3)
            plt.title("VARI")
            plt.imshow(vari, cmap='RdYlGn')
            plt.colorbar()

            plt.subplot(1, 5, 4)
            plt.title("GLI")
            plt.imshow(gli, cmap='RdYlGn')
            plt.colorbar()

            plt.subplot(1, 5, 5)
            plt.title("ExG")
            plt.imshow(exg, cmap='RdYlGn')
            plt.colorbar()

            plt.savefig(output_path.replace(".jpg", "_indices.png"))
            plt.close()

            print(f"Processed image: {img_name}")

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
