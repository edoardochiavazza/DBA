import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as stats


def load_and_resize_image(image_path, target_size=(100, 300), mode='gray'):
    """
    Load and resize an image to the specified dimensions.

    Args:
        image_path (str): Path to the input image
        target_size (tuple): Target size as (height, width)
        mode (str): The desired image format, either 'gray' or 'RGB'

    Returns:
        torch.Tensor: Resized image tensor with shape (3, height, width) and values in [0,1]
    """
    # Load the image using torchvision
    try:
        img = read_image(image_path)
        # Convert to float and normalize to [0,1] if needed
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
    except Exception as e:
        print(f"Error loading image with torchvision: {e}")
        # Fallback to PIL
        img = Image.open(image_path)
        img = transforms.ToTensor()(img)  # ToTensor automatically normalizes to [0,1]

    if mode == 'RGB':
        # Make sure image has 3 channels (RGB)
        if img.shape[0] == 1:  # Convert grayscale to RGB
            img = img.repeat(3, 1, 1)
        elif img.shape[0] > 3:  # Keep only RGB channels (drop alpha)
            img = img[:3, :, :]
    if mode == 'gray':
        if img.shape[0] != 1:
            gray_transform = transforms.Grayscale(num_output_channels=1)
            img = gray_transform(img)

    # Resize image to target size
    resize_transform = transforms.Resize(target_size)
    img_resized = resize_transform(img)

    # Ensure values are in [0,1] range
    img_resized = torch.clamp(img_resized, 0.0, 1.0)

    return img_resized


def compute_grid_cells(img_tensor, grid_size=(10, 10)):
    """
    Partition an image into a grid of cells.

    Args:
        img_tensor (torch.Tensor): Input image tensor with shape (3, height, width)
        grid_size (tuple): Grid dimensions as (rows, columns)

    Returns:
        list: List of image cells, each as a torch.Tensor
        tuple: Cell dimensions as (cell_height, cell_width)
    """
    _, height, width = img_tensor.shape
    grid_rows, grid_cols = grid_size

    # Calculate grid cell dimensions
    cell_height = height // grid_rows
    cell_width = width // grid_cols

    # Process each grid cell
    grid_cells = []
    for i in range(grid_rows):  # rows
        for j in range(grid_cols):  # columns
            # Extract grid cell
            y_start = i * cell_height
            y_end = (i + 1) * cell_height
            x_start = j * cell_width
            x_end = (j + 1) * cell_width

            cell = img_tensor[:, y_start:y_end, x_start:x_end]
            grid_cells.append(cell)

    return grid_cells, (cell_height, cell_width)


def compute_color_moments(grid_cells, grid_size=(10, 10), image_name=None, visualize=True, output_dir="Task1/result"):
    """
    Compute color moments (mean, std, skewness) for each RGB channel in each grid cell.

    Args:
        grid_cells (list): List of image cells as torch.Tensors
        grid_size (tuple): Grid dimensions as (rows, columns)
        image_name (str): Name of the image for saving outputs
        visualize (bool): Whether to generate visualizations
        output_dir (str): Directory to save outputs

    Returns:
        np.ndarray: Flattened feature vector with color moments
    """
    grid_rows, grid_cols = grid_size

    # Create output directory if needed
    if image_name and (visualize or output_dir):
        full_output_dir = os.path.join(output_dir, image_name) if image_name else output_dir
        os.makedirs(full_output_dir, exist_ok=True)

    # Initialize feature vector: grid_rows × grid_cols × 3 channels × 3 moments
    feature_vector = np.zeros((grid_rows, grid_cols, 3, 3))

    # Process each grid cell
    cell_idx = 0
    for i in range(grid_rows):
        for j in range(grid_cols):
            cell = grid_cells[cell_idx]
            cell_np = cell.numpy()

            for channel in range(3):  # RGB channels
                # Mean (1st moment)
                mean = np.mean(cell_np[channel])
                feature_vector[i, j, channel, 0] = mean

                # Standard deviation (2nd moment)
                std = np.std(cell_np[channel])
                feature_vector[i, j, channel, 1] = std

                # Skewness (3rd moment)
                if std == 0:
                    skewness = 0
                else:
                    skewness = stats.skew(cell_np[channel].flatten())
                feature_vector[i, j, channel, 2] = skewness

            cell_idx += 1

    # Visualize if requested
    if visualize and image_name:
        # We need the original resized image for visualization
        # Reconstruct it from grid cells (this is a bit hacky, but works for visualization)
        img_resized = reconstruct_image_from_cells(grid_cells, grid_size)
        visualize_color_moments(img_resized, feature_vector, full_output_dir)

    # Flatten the feature vector
    flattened_features = feature_vector.flatten()

    # Save feature vector if output directory is specified
    if image_name and output_dir:
        np.save(os.path.join(full_output_dir, "cm10x10_features.npy"), flattened_features)

    return flattened_features


def reconstruct_image_from_cells(grid_cells, grid_size):
    """
    Reconstruct the original image from grid cells (for visualization purposes).

    Args:
        grid_cells (list): List of image cells as torch.Tensors
        grid_size (tuple): Grid dimensions as (rows, columns)

    Returns:
        torch.Tensor: Reconstructed image tensor
    """
    grid_rows, grid_cols = grid_size

    # Get dimensions from first cell
    channels, cell_height, cell_width = grid_cells[0].shape

    # Initialize reconstructed image
    img_height = grid_rows * cell_height
    img_width = grid_cols * cell_width
    reconstructed = torch.zeros(channels, img_height, img_width)

    # Place each cell back
    cell_idx = 0
    for i in range(grid_rows):
        for j in range(grid_cols):
            y_start = i * cell_height
            y_end = (i + 1) * cell_height
            x_start = j * cell_width
            x_end = (j + 1) * cell_width

            reconstructed[:, y_start:y_end, x_start:x_end] = grid_cells[cell_idx]
            cell_idx += 1

    return reconstructed


def visualize_color_moments(img_tensor, feature_vector, output_dir):
    """
    Create visualizations of the grid and color moments.

    Args:
        img_tensor (torch.Tensor): Original resized image
        feature_vector (np.ndarray): Color moments feature vector
        output_dir (str): Directory to save visualizations
    """
    # Convert tensor to numpy for matplotlib and ensure proper range [0,1]
    img_np = img_tensor.permute(1, 2, 0).numpy()

    # Ensure the image values are in the correct range [0,1] for matplotlib
    if img_np.max() > 1.0:
        img_np = img_np / 255.0

    # Clip values to [0,1] range to avoid matplotlib warnings
    img_np = np.clip(img_np, 0.0, 1.0)

    # Create figure for grid visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.imshow(img_np)

    # Draw grid lines
    grid_rows, grid_cols = 10, 10
    height, width = img_np.shape[:2]
    cell_height = height // grid_rows
    cell_width = width // grid_cols

    for i in range(grid_rows + 1):
        ax.axhline(i * cell_height, color='red', linestyle='-', linewidth=0.5)
    for j in range(grid_cols + 1):
        ax.axvline(j * cell_width, color='red', linestyle='-', linewidth=0.5)

    ax.set_title(f"{width}×{height} Image with {grid_cols}×{grid_rows} Grid")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    plt.tight_layout()

    # Save the grid visualization
    grid_vis_path = os.path.join(output_dir, "grid_visualization.png")
    plt.savefig(grid_vis_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create heatmap visualizations for each moment across the grid
    channel_names = ['Red', 'Green', 'Blue']
    moment_names = ['Mean', 'Standard Deviation', 'Skewness']

    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for channel in range(3):  # Rows
        for moment in range(3):  # Columns
            # Extract the specific moment for this channel across all cells
            moment_values = feature_vector[:, :, channel, moment]

            # Create heatmap
            im = axes[channel, moment].imshow(moment_values, cmap='viridis')

            # Set title
            axes[channel, moment].set_title(f"{channel_names[channel]} - {moment_names[moment]}")

            # Add colorbar
            plt.colorbar(im, ax=axes[channel, moment])

            # Add grid lines
            for i in range(grid_rows + 1):
                axes[channel, moment].axhline(i - 0.5, color='white', linestyle='-', linewidth=0.5)
                axes[channel, moment].axvline(i - 0.5, color='white', linestyle='-', linewidth=0.5)

            # Set axis labels
            axes[channel, moment].set_xlabel("Grid Column")
            axes[channel, moment].set_ylabel("Grid Row")

            # Add value annotations
            for i in range(grid_rows):
                for j in range(grid_cols):
                    val = moment_values[i, j]
                    # Determine text color for better visibility
                    text_color = 'white' if val < np.max(moment_values) * 0.5 else 'black'
                    axes[channel, moment].text(j, i, f"{val:.2f}",
                                               ha="center", va="center",
                                               color=text_color, fontsize=6)

    plt.tight_layout()

    # Save the combined heatmap
    heatmap_path = os.path.join(output_dir, "color_moments_heatmap_3x3.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Grid visualization saved to {grid_vis_path}")
    print(f"Color moments heatmap saved to {heatmap_path}")


def analyze_feature_vector(feature_vector, feature_name="Feature"):
    """
    Analyze and print statistics about any feature vector.

    Args:
        feature_vector (np.ndarray): Feature vector to analyze
        feature_name (str): Name of the feature type for display
    """
    print(f"\n{feature_name.upper()} Feature Vector Statistics:")
    print("=" * 50)
    print(f"Feature vector shape: {feature_vector.shape}")
    print(f"Feature vector length: {len(feature_vector)}")
    print(f"Mean: {np.mean(feature_vector):.4f}")
    print(f"Standard deviation: {np.std(feature_vector):.4f}")
    print(f"Min value: {np.min(feature_vector):.4f}")
    print(f"Max value: {np.max(feature_vector):.4f}")

    # Detailed analysis for color moments (if the shape matches)
    if feature_name == 'color_moments':  # 10x10x3x3 = 900 (likely color moments)
        try:
            reshaped = feature_vector.reshape(10, 10, 3, 3)

            # Calculate statistics per channel and moment
            mean_values = np.mean(reshaped, axis=(0, 1))
            std_values = np.std(reshaped, axis=(0, 1))
            min_values = np.min(reshaped, axis=(0, 1))
            max_values = np.max(reshaped, axis=(0, 1))

            moment_names = ["Mean", "Std Dev", "Skewness"]
            channel_names = ["Red", "Green", "Blue"]

            print("\nDetailed Channel Analysis (Color Moments):")
            print("-" * 40)
            for c in range(3):
                print(f"\n{channel_names[c]} Channel:")
                for m in range(3):
                    print(f"  {moment_names[m]}:")
                    print(f"    - Average across grid: {mean_values[c, m]:.4f}")
                    print(f"    - Standard deviation: {std_values[c, m]:.4f}")
                    print(f"    - Range: [{min_values[c, m]:.4f}, {max_values[c, m]:.4f}]")
        except:
            pass  # If reshape fails, skip detailed analysis


# Main processing function for any grid-based features
def process_image_features(image_path, feature_functions=None, visualize=True, output_dir="results", mode='gray'):
    """
    Main function to process an image and extract features using grid-based methods.

    Args:
        image_path (str): Path to the input image
        feature_functions (dict): Dictionary of feature functions to apply
                                 Format: {'function_name': {'func': function, 'params': dict}}
        visualize (bool): Whether to generate visualizations
        output_dir (str): Directory to save outputs
        mode (str): RGB or gray for elaborate the image

    Returns:
        dict: Dictionary containing all extracted features

    """
    # Extract image name
    print(f"Processing image: {mode}")
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"Processing image: {image_name}")

    # Create output directory
    full_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(full_output_dir, exist_ok=True)

    # Load and resize image
    img_tensor = load_and_resize_image(image_path, target_size=(100, 300), mode=mode)

    # Compute grid cells (shared by all grid-based features)
    print("Computing grid cells...")
    grid_cells, cell_dims = compute_grid_cells(img_tensor, grid_size=(10, 10))

    # Store results
    results = {}
    # Apply each feature function
    for feature_name, feature_config in feature_functions.items():
        features = ''
        print(f"Computing {feature_name} features...")

        func = feature_config['func']
        params = feature_config.get('params', {})

        # Add common parameters
        params['image_name'] = image_name

        # Execute the feature function
        if feature_name == 'color_moments' and params["performed"]:
            features = func(grid_cells, **params)
        if feature_name == 'hog_features' and params["performed"]:
            features = func(grid_cells, **params)

        results[feature_name] = features

        # Analyze the features
        # analyze_feature_vector(features, feature_name)

    print(f"\nFeature extraction complete for {image_name}")
    return results


def process_image_hog(image_path, visualize=True, output_dir="results"):
    results = process_image_features(
        image_path=image_path,
        feature_functions={
            'hog_features': {
                'func': compute_hog_features,
                'params': {'grid_size': (10, 10), 'output_dir': output_dir, 'performed': True, 'mode': 'gray'}
            }
        },
        visualize=visualize,
        output_dir=output_dir
    )

    return results['hog_features']


# Convenience function for color moments only
def process_image_color_moments(image_path, visualize=True, output_dir="results"):
    """
    Convenience function to process an image and extract only color moments features.

    Args:
        image_path (str): Path to the input image
        visualize (bool): Whether to generate visualizations
        output_dir (str): Directory to save outputs

    Returns:
        np.ndarray: Color moments feature vector
    """
    results = process_image_features(
        image_path=image_path,
        feature_functions={
            'color_moments': {
                'func': compute_color_moments,
                'params': {'grid_size': (10, 10), 'visualize': visualize, 'output_dir': output_dir, 'performed': False}
            }
        },
        visualize=visualize,
        output_dir=output_dir
    )

    return results['color_moments']


def compute_gradient(cell):
    kernel_x = np.array([[-1, 0, 1]], dtype=np.float32)  # dI/dx
    kernel_y = np.array([[-1], [0], [1]], dtype=np.float32)  # dI/dy



    for i in range(cell.shape[1]):
        for j in range(cell.shape[2]):
            grad_x = sum(cell[j:j + 3] * kernel_x)
            grad_y = sum(cell[i:i + 3] * kernel_y)
            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            orientation = np.arctan2(grad_y, grad_x)
            print(i, j, magnitude, orientation)


def compute_hog_features(grid_cells, grid_size=(10, 10), image_name=None, output_dir="results", **kwargs):
    """
    Template function for computing custom features from grid cells.

    Args:
        grid_cells (list): List of image cells as torch.Tensors
        grid_size (tuple): Grid dimensions as (rows, columns)
        image_name (str): Name of the image for saving outputs
        output_dir (str): Directory to save outputs
        **kwargs: Additional parameters

    Returns:
        np.ndarray: Feature vector
    """
    # Example: compute average brightness per cell
    grid_rows, grid_cols = grid_size

    # Maschere per i gradient

    cell_idx = 0
    print(len(grid_cells))
    for i in range(grid_rows):
        for j in range(grid_cols):
            cell = grid_cells[cell_idx]
            cell = np.pad(cell, 1, 'edge')
            compute_gradient(cell)

    '''
    # Save feature vector if needed
    if image_name and output_dir:
        np.save(os.path.join(full_output_dir, "example_features.npy"), feature_vector)
    '''
    return


# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_p = "Part1/brain_glioma/brain_glioma_0001.jpg"
    hog_features = process_image_hog(
        image_path=image_p,
        visualize=True,
        output_dir="results"
    )
    '''
    # Method 1: Extract only color moments (simple)
    print("=== METHOD 1: Color Moments Only ===")
    cm_features = process_image_color_moments(
        image_path=image_path,
        visualize=True,
        output_dir="results"
    )
    print(f"Color Moments features shape: {cm_features.shape}")

    print("\n" + "=" * 60 + "\n")

    # Method 2: Extract multiple features using the general function
    print("=== METHOD 2: Multiple Features ===")
    feature_functions = {
        'color_moments': {
            'func': compute_color_moments,
            'params': {'grid_size': (10, 10), 'visualize': True, 'output_dir': "results"}
        },
        'example_feature': {
            'func': compute_example_feature,
            'params': {'grid_size': (10, 10), 'output_dir': "results"}
        }
    }

    results = process_image_features(
        image_path=image_path,
        feature_functions=feature_functions,
        visualize=True,
        output_dir="results"
    )

    print(f"\nExtracted features:")
    for feature_name, features in results.items():
        print(f"- {feature_name}: {features.shape}")

    print("\n=== How to add new features ===")
    print("1. Create a function following the template 'compute_example_feature'")
    print("2. Add it to the feature_functions dictionary")
    print("3. The function will automatically receive grid_cells and other parameters")
'''
