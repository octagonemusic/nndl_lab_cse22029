import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D  # type: ignore

def load_and_preprocess_image(image_path):
    """Load and preprocess the image for the model."""
    img = load_img(image_path, target_size=(64, 64), color_mode='grayscale')
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

def get_feature_maps(model, image):
    """Get feature maps from the model for the given image."""
    # Call the model with a dummy input to ensure it is built
    model(np.zeros((1, 64, 64, 1)))  # Dummy input to build the model
    layer_outputs = [layer.output for layer in model.layers if isinstance(layer, (Conv2D, MaxPooling2D))]
    # Use model.inputs instead of model.input
    feature_map_model = Model(inputs=model.inputs, outputs=layer_outputs)
    feature_maps = feature_map_model.predict(image)
    return feature_maps

def plot_feature_maps(feature_maps, save_dir):
    """Plot and save the feature maps."""
    for layer_idx, feature_map in enumerate(feature_maps):
        num_filters = feature_map.shape[-1]
        size = feature_map.shape[1]
        
        # Create a grid of subplots
        plt.figure(figsize=(15, 15))
        for i in range(num_filters):
            plt.subplot(num_filters // 8 + 1, 8, i + 1)  # Adjust grid size as needed
            plt.imshow(feature_map[0, :, :, i], cmap='viridis')
            plt.axis('off')
        
        plt.suptitle(f'Feature Maps from Layer {layer_idx + 1}', fontsize=16)
        
        # Save the figure
        plt.savefig(os.path.join(save_dir, f'feature_maps_layer_{layer_idx + 1}.png'))
        plt.close()  # Close the figure to free memory

def main():
    # Load the trained model from A3.py
    model = load_model('asl_classifier.keras')  # Ensure this is the correct path to your model

    # Specify the path to an image from your dataset
    dataset_path = 'dataset'
    letter = 'A'  # Change this to the letter you want to visualize
    img_num = 0  # Change this to the image number you want to visualize
    image_path = os.path.join(dataset_path, f"{letter}-samples", f"{img_num}.jpg")

    # Load and preprocess the image
    image = load_and_preprocess_image(image_path)

    # Create a directory to save feature maps
    save_dir = 'feature_maps'
    os.makedirs(save_dir, exist_ok=True)

    # Get feature maps
    feature_maps = get_feature_maps(model, image)

    # Plot and save feature maps
    plot_feature_maps(feature_maps, save_dir)

if __name__ == "__main__":
    main()