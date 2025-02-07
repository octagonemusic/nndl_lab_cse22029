import numpy as np
from scipy import signal
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

def load_and_preprocess_image(path):
    """
    Loads and preprocesses the image.
    Args:
        path (str): Path to the image file
    Returns:
        ndarray: Preprocessed image region
    """
    # Load and convert to grayscale if needed
    img = imread(path)
    if len(img.shape) == 3:
        img = rgb2gray(img)
    
    # Let's modify the ROI to include the full text
    # Remove the cropping entirely
    return img  # Return full image instead of img[40:350, 20:350]

def define_filters():
    """
    Defines the three filters for image processing.
    Returns:
        tuple: (edge_detector, gaussian_like, box_filter)
    """
    # Edge detection filter (stronger)
    fil1 = np.array([
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0]
    ]) * 2  # Multiplied by 2 to make edges more prominent
    
    # Gaussian-like smoothing filter (stronger)
    fil2 = np.array([
        [0.1, 0.2, 0.1],
        [0.2, 0.8, 0.2],
        [0.1, 0.2, 0.1]
    ])
    
    # Larger box filter for more visible smoothing
    fil3 = np.ones((7, 7)) * (1/49)  # 7x7 filter instead of 5x5
    
    return fil1, fil2, fil3

def apply_filters(image, filters):
    """
    Applies convolution with all three filters.
    Args:
        image (ndarray): Input image
        filters (tuple): Three filters to apply
    Returns:
        tuple: Three filtered images
    """
    fil1, fil2, fil3 = filters
    
    # Apply convolutions with symmetric boundary conditions
    grad1 = signal.convolve2d(image, fil1, boundary='symm', mode='same')
    grad2 = signal.convolve2d(image, fil2, boundary='symm', mode='same')
    grad3 = signal.convolve2d(image, fil3, boundary='symm', mode='same')
    
    return grad1, grad2, grad3

def plot_results(original, filtered_images):
    """
    Creates and saves plots of original and filtered images.
    """
    grad1, grad2, grad3 = filtered_images
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Edge detection result
    axes[0, 1].imshow(np.abs(grad1), cmap='gray')
    axes[0, 1].set_title('Edge Detection\n(Highlights sharp changes)')
    axes[0, 1].axis('off')
    
    # Gaussian-like smoothing result
    axes[1, 0].imshow(grad2, cmap='gray')
    axes[1, 0].set_title('Gaussian Smoothing\n(Subtle noise reduction)')
    axes[1, 0].axis('off')
    
    # Box filter result
    axes[1, 1].imshow(grad3, cmap='gray')
    axes[1, 1].set_title('Box Filter\n(Strong blurring)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('filtered_images.png', dpi=300)
    plt.close()

def main():
    """
    Main function to orchestrate the image filtering process.
    """
    # Load and preprocess image
    image = load_and_preprocess_image('lab-3/Neural.JPG')
    
    # Get filters
    filters = define_filters()
    
    # Apply filters
    filtered_images = apply_filters(image, filters)
    
    # Display results
    plot_results(image, filtered_images)

if __name__ == "__main__":
    main()