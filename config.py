import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'archive')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
BATCH_SIZE = 5  # Adjust based on your system's capability
CONFIDENCE_THRESHOLD = 0.5

# Classes for the dataset
CLASSES = ['Gall Midge', 'Bacterial Canker', 'Powdery Mildew', 'Healthy',
           'Anthracnose', 'Sooty Mould', 'Cutting Weevil', 'Die Back']
