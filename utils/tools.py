import matplotlib.pyplot as plt
import numpy as np

def visualize_binary_image(image, title=None):
    """
    Function to visualize a binary image

    Parameters:
    - image (numpy array): Binary image, values should be 0 or 1
    - title (str, optional): Title of the image
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
