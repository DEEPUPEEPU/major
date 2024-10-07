import numpy as np

# Function to calculate NDVI
def calculate_ndvi(image):
    red = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float)
    ndvi = (green - red) / (green + red + 1e-5)  # Avoid division by zero
    return ndvi

# Function to calculate LNDVI
def calculate_lndvi(image):
    red = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float)
    blue = image[:, :, 2].astype(float)
    lndvi = (green - blue) / (green + blue + 1e-5)  # Avoid division by zero
    return lndvi

# Function to calculate VARI
def calculate_vari(image):
    red = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float)
    blue = image[:, :, 2].astype(float)
    vari = (green - red) / (green + red - blue + 1e-5)  # Avoid division by zero
    return vari

# Function to calculate GLI
def calculate_gli(image):
    red = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float)
    blue = image[:, :, 2].astype(float)
    gli = (2 * green - red - blue) / (2 * green + red + blue + 1e-5)  # Avoid division by zero
    return gli

# Function to calculate ExG
def calculate_exg(image):
    red = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float)
    blue = image[:, :, 2].astype(float)
    exg = (2 * green - red - blue) / (green + red + blue + 1e-5)  # Avoid division by zero
    return exg
