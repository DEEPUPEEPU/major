import numpy as np

def calculate_ndvi(image):
    red = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float)
    ndvi = (green - red) / (green + red + 1e-5)
    return ndvi

def calculate_vari(image):
    red = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float)
    blue = image[:, :, 2].astype(float)
    vari = (green - red) / (green + red - blue + 1e-5)
    return vari

def calculate_gli(image):
    red = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float)
    blue = image[:, :, 2].astype(float)
    gli = (2 * green - red - blue) / (2 * green + red + blue + 1e-5)
    return gli

def calculate_exg(image):
    red = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float)
    blue = image[:, :, 2].astype(float)
    exg = 2 * green - red - blue
    return exg
