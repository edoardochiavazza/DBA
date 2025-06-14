import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.io import read_image


class ResNetFeatureExtractor:
    """ResNet-based feature extractor for deep learning features"""

    def __init__(self, model_name='resnet50', pretrained=True):
        """
        Initialize ResNet feature extractor

        Args:
            model_name: 'resnet50'
            pretrained: Use pretrained weights
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load pretrained ResNet
        model_dict = {
            'resnet50': models.resnet50,
        }

        if model_name not in model_dict:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model = model_dict[model_name](pretrained=pretrained)
        self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Storage for hooked features
        self.hooked_features = {}

    def _hook_fn(self, name):
        """Create hook function for layer"""

        def hook(module, input, output):
            self.hooked_features[name] = output.detach()

        return hook

    def _preprocess_image(self, image_tensor):
        """Preprocess image tensor for ResNet"""
        # Convert to PIL Image first for consistent preprocessing
        if image_tensor.shape[0] == 1:  # Grayscale
            image_tensor = image_tensor.repeat(3, 1, 1)  # Convert to RGB
        elif image_tensor.shape[0] > 3:
            image_tensor = image_tensor[:3, :, :]  # Keep only RGB

        # Convert tensor to PIL Image
        img_pil = transforms.ToPILImage()(image_tensor)

        # Apply standard ImageNet preprocessing
        processed = self.transform(img_pil).unsqueeze(0)
        return processed.to(self.device)

    def extract_avgpool_1024(self, image_tensor):
        """ResNet-AvgPool-1024: Extract from avgpool layer and reduce to 1024D"""
        hook = self.model.avgpool.register_forward_hook(self._hook_fn('avgpool'))

        try:
            input_tensor = self._preprocess_image(image_tensor)

            with torch.no_grad():
                _ = self.model(input_tensor)

            avgpool_features = self.hooked_features['avgpool'].squeeze()
            features_np = avgpool_features.cpu().numpy()

            # Reduce 2048D to 1024D by averaging consecutive pairs
            if len(features_np) >= 2048:
                features_2048 = features_np[:2048]
                reduced_features = features_2048.reshape(-1, 2).mean(axis=1)
            else:
                # Pad or truncate to 1024D
                reduced_features = np.pad(features_np, (0, max(0, 1024 - len(features_np))))[:1024]

            return reduced_features

        finally:
            hook.remove()
            self.hooked_features.clear()

    def extract_layer3_1024(self, image_tensor):
        """ResNet-Layer3-1024: Extract from layer3 and global average pool"""
        hook = self.model.layer3.register_forward_hook(self._hook_fn('layer3'))

        try:
            input_tensor = self._preprocess_image(image_tensor)

            with torch.no_grad():
                _ = self.model(input_tensor)

            layer3_features = self.hooked_features['layer3'].squeeze()

            # Global average pooling
            if len(layer3_features.shape) == 3:
                pooled_features = torch.mean(layer3_features, dim=(1, 2))
            else:
                pooled_features = layer3_features

            features_np = pooled_features.cpu().numpy()

            # Ensure exactly 1024 dimensions
            if len(features_np) > 1024:
                features_np = features_np[:1024]
            elif len(features_np) < 1024:
                features_np = np.pad(features_np, (0, 1024 - len(features_np)))

            return features_np

        finally:
            hook.remove()
            self.hooked_features.clear()

    def extract_fc_1000(self, image_tensor):
        """ResNet-FC-1000: Extract from final FC layer"""
        hook = self.model.fc.register_forward_hook(self._hook_fn('fc'))

        try:
            input_tensor = self._preprocess_image(image_tensor)

            with torch.no_grad():
                _ = self.model(input_tensor)

            fc_features = self.hooked_features['fc'].squeeze()
            features_np = fc_features.cpu().numpy()

            return features_np

        finally:
            hook.remove()
            self.hooked_features.clear()


def load_and_resize_image(image_path, target_size=(100, 300), mode='RGB'):
    """
    Load and resize an image to the specified dimensions.

    Args:
        image_path (str): Path to the input image
        target_size (tuple): Target size as (height, width)
        mode (str): The desired image format, either 'gray' or 'RGB'

    Returns:
        torch.Tensor: Resized image tensor with shape (C, height, width) and values in [0,1]
    """
    try:
        img = read_image(image_path)
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
    except Exception as e:
        print(f"Error loading image with torchvision: {e}")
        img = Image.open(image_path)
        img = transforms.ToTensor()(img)

    # Handle color mode conversion
    if mode == 'RGB':
        if img.shape[0] == 1:  # Grayscale to RGB
            img = img.repeat(3, 1, 1)
        elif img.shape[0] > 3:  # Keep only RGB channels
            img = img[:3, :, :]
    elif mode == 'gray':
        if img.shape[0] != 1:
            gray_transform = transforms.Grayscale(num_output_channels=1)
            img = gray_transform(img)

    # Resize image
    resize_transform = transforms.Resize(target_size)
    img_resized = resize_transform(img)
    img_resized = torch.clamp(img_resized, 0.0, 1.0)

    return img_resized


def compute_grid_cells(img_tensor, grid_size=(10, 10)):
    """
    Partition an image into a grid of cells.

    Args:
        img_tensor (torch.Tensor): Input image tensor with shape (C, height, width)
        grid_size (tuple): Grid dimensions as (rows, columns)

    Returns:
        list: List of image cells, each as a torch.Tensor
        tuple: Cell dimensions as (cell_height, cell_width)
    """
    _, height, width = img_tensor.shape
    grid_rows, grid_cols = grid_size

    cell_height = height // grid_rows
    cell_width = width // grid_cols

    grid_cells = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            y_start = i * cell_height
            y_end = (i + 1) * cell_height
            x_start = j * cell_width
            x_end = (j + 1) * cell_width

            cell = img_tensor[:, y_start:y_end, x_start:x_end]
            grid_cells.append(cell)

    return grid_cells, (cell_height, cell_width)


def compute_color_moments(grid_cells, grid_size=(10, 10), image_name=None,
                          visualize=False, output_dir="Task1/results", **kwargs):
    """
    Compute color moments (mean, std, skewness) for each RGB channel in each grid cell.
    """
    grid_rows, grid_cols = grid_size

    if image_name and output_dir:
        full_output_dir = os.path.join(output_dir, image_name)
        os.makedirs(full_output_dir, exist_ok=True)

    feature_vector = np.zeros((grid_rows, grid_cols, 3, 3))

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

    if visualize and image_name:
        img_resized = reconstruct_image_from_cells(grid_cells, grid_size)
        visualize_color_moments(img_resized, feature_vector, full_output_dir)

    flattened_features = feature_vector.flatten()

    if image_name and output_dir:
        np.save(os.path.join(full_output_dir, "cm10x10_features.npy"), flattened_features)

    return flattened_features


def compute_gradient(cell):
    """Compute HOG gradient histogram for a cell"""
    kernel_x = np.array([[-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1], [0], [1]], dtype=np.float32)
    histogram = np.zeros(9, dtype=np.float32)

    image = cell[0]  # Use first channel for grayscale

    for row in range(1, image.shape[0] - 1):
        for col in range(1, image.shape[1] - 1):
            region_x = image[row, col - 1:col + 2]
            grad_x = np.sum(region_x * kernel_x)

            region_y = image[row - 1:row + 2, col]
            grad_y = np.sum(region_y * kernel_y[:, 0])

            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            orientation_deg = (np.degrees(np.arctan2(grad_y, grad_x)) + 360) % 360
            bin_idx = int(orientation_deg // 40)
            histogram[bin_idx] += magnitude

    return histogram


def compute_hog_features(grid_cells, grid_size=(10, 10), image_name=None,
                         output_dir="Task1/results", **kwargs):
    """
    Compute HOG features from grid cells.
    """
    features_descriptor = []

    for cell in grid_cells:
        # Add padding for gradient computation
        cell_padded = np.pad(cell, pad_width=((0, 0), (1, 1), (1, 1)), mode='edge')
        histogram = compute_gradient(cell_padded)
        features_descriptor.append(histogram)

    features_descriptor = np.concatenate(features_descriptor)

    if image_name and output_dir:
        full_output_dir = os.path.join(output_dir, image_name)
        os.makedirs(full_output_dir, exist_ok=True)
        np.save(os.path.join(full_output_dir, "hog_features.npy"), features_descriptor)

    return features_descriptor


def compute_resnet_features(img_tensor, extractor_type='avgpool_1024', image_name=None,
                            output_dir="Task1/results", model_name='resnet50', **kwargs):
    """
    Compute ResNet features using the specified extraction method.

    Args:
        img_tensor (torch.Tensor): Input image tensor
        extractor_type (str): Type of extraction ('avgpool_1024', 'layer3_1024', 'fc_1000')
        image_name (str): Name for saving features
        output_dir (str): Output directory
        model_name (str): ResNet model to use

    Returns:
        np.ndarray: Extracted features
    """
    # Initialize extractor (you might want to cache this for efficiency)
    extractor = ResNetFeatureExtractor(model_name=model_name)

    # Extract features based on type
    if extractor_type == 'avgpool_1024':
        features = extractor.extract_avgpool_1024(img_tensor)
    elif extractor_type == 'layer3_1024':
        features = extractor.extract_layer3_1024(img_tensor)
    elif extractor_type == 'fc_1000':
        features = extractor.extract_fc_1000(img_tensor)
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")

    # Save features if requested
    if image_name and output_dir:
        full_output_dir = os.path.join(output_dir, image_name)
        os.makedirs(full_output_dir, exist_ok=True)
        feature_filename = f"resnet_{extractor_type}_features.npy"
        np.save(os.path.join(full_output_dir, feature_filename), features)

    return features


def reconstruct_image_from_cells(grid_cells, grid_size):
    """Reconstruct image from grid cells for visualization."""
    grid_rows, grid_cols = grid_size
    channels, cell_height, cell_width = grid_cells[0].shape

    img_height = grid_rows * cell_height
    img_width = grid_cols * cell_width
    reconstructed = torch.zeros(channels, img_height, img_width)

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
    """Create visualizations of the grid and color moments."""
    img_np = img_tensor.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0.0, 1.0)

    # Grid visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.imshow(img_np)

    grid_rows, grid_cols = 10, 10
    height, width = img_np.shape[:2]
    cell_height = height // grid_rows
    cell_width = width // grid_cols

    for i in range(grid_rows + 1):
        ax.axhline(i * cell_height, color='red', linestyle='-', linewidth=0.5)
    for j in range(grid_cols + 1):
        ax.axvline(j * cell_width, color='red', linestyle='-', linewidth=0.5)

    ax.set_title(f"{width}×{height} Image with {grid_cols}×{grid_rows} Grid")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grid_visualization.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Heatmap visualization
    channel_names = ['Red', 'Green', 'Blue']
    moment_names = ['Mean', 'Standard Deviation', 'Skewness']

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for channel in range(3):
        for moment in range(3):
            moment_values = feature_vector[:, :, channel, moment]
            im = axes[channel, moment].imshow(moment_values, cmap='viridis')
            axes[channel, moment].set_title(f"{channel_names[channel]} - {moment_names[moment]}")
            plt.colorbar(im, ax=axes[channel, moment])

            for i in range(grid_rows):
                for j in range(grid_cols):
                    val = moment_values[i, j]
                    text_color = 'white' if val < np.max(moment_values) * 0.5 else 'black'
                    axes[channel, moment].text(j, i, f"{val:.2f}",
                                               ha="center", va="center",
                                               color=text_color, fontsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "color_moments_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()


def analyze_feature_vector(feature_vector, feature_name="Feature"):
    """Analyze and print statistics about any feature vector."""
    print(f"\n{feature_name.upper()} Feature Vector Statistics:")
    print("=" * 50)
    print(f"Shape: {feature_vector.shape}")
    print(f"Length: {len(feature_vector)}")
    print(f"Mean: {np.mean(feature_vector):.4f}")
    print(f"Std: {np.std(feature_vector):.4f}")
    print(f"Min: {np.min(feature_vector):.4f}")
    print(f"Max: {np.max(feature_vector):.4f}")


def process_image_with_features(image_path, feature_configs, visualize=False, output_dir="Task1/results"):
    """
    Main function to process an image and extract multiple types of features.

    Args:
        image_path (str): Path to the input image
        feature_configs (dict): Dictionary of feature configurations
        visualize (bool): Whether to generate visualizations
        output_dir (str): Directory to save outputs

    Returns:
        dict: Dictionary containing all extracted features
    """
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"Processing image: {image_name}")

    full_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(full_output_dir, exist_ok=True)

    results = {}

    for feature_name, config in feature_configs.items():
        if not config.get('enabled', True):
            continue

        print(f"Computing {feature_name} features...")

        # Load image with appropriate mode
        mode = config.get('mode', 'RGB')
        target_size = config.get('target_size', (100, 300))

        if feature_name.startswith('resnet'):
            # ResNet features work on full image
            img_tensor = load_and_resize_image(image_path, target_size=(224, 224), mode='RGB')
            features = compute_resnet_features(
                img_tensor=img_tensor,
                extractor_type=config['extractor_type'],
                image_name=image_name,
                output_dir=output_dir,
                model_name=config.get('model_name', 'resnet50')
            )
        else:
            # Grid-based features
            img_tensor = load_and_resize_image(image_path, target_size=target_size, mode=mode)
            grid_cells, _ = compute_grid_cells(img_tensor, grid_size=config.get('grid_size', (10, 10)))

            func = config['func']
            params = config.get('params', {})
            params.update({
                'image_name': image_name,
                'output_dir': output_dir,
                'visualize': visualize
            })

            features = func(grid_cells, **params)

        results[feature_name] = features
        #analyze_feature_vector(features, feature_name)

    print(f"\nFeature extraction complete for {image_name}")
    return results


# Convenience functions for specific feature types
def process_image_color_moments(image_path, visualize=False, output_dir="Task1/results"):
    """Extract only color moments features."""
    return process_image_with_features(
        image_path=image_path,
        feature_configs={
            'color_moments': {
                'func': compute_color_moments,
                'mode': 'RGB',
                'target_size': (100, 300),
                'grid_size': (10, 10),
                'enabled': True
            }
        },
        visualize=visualize,
        output_dir=output_dir
    )['color_moments']


def process_image_hog(image_path, visualize=False, output_dir="Task1/results"):
    """Extract only HOG features."""
    return process_image_with_features(
        image_path=image_path,
        feature_configs={
            'hog_features': {
                'func': compute_hog_features,
                'mode': 'gray',
                'target_size': (100, 300),
                'grid_size': (10, 10),
                'enabled': True
            }
        },
        visualize=visualize,
        output_dir=output_dir
    )['hog_features']


def process_image_resnet(image_path, extractor_type='avgpool_1024', model_name='resnet50',
                         output_dir="results"):
    """Extract ResNet features."""
    return process_image_with_features(
        image_path=image_path,
        feature_configs={
            f'resnet_{extractor_type}': {
                'extractor_type': extractor_type,
                'model_name': model_name,
                'enabled': True
            }
        },
        visualize=False,
        output_dir=output_dir
    )[f'resnet_{extractor_type}']


def process_image_all_features(image_path, output_dir="Task1/results"):
    """Extract all types of features from an image."""
    return process_image_with_features(
        image_path=image_path,
        feature_configs={
            'color_moments': {
                'func': compute_color_moments,
                'mode': 'RGB',
                'target_size': (100, 300),
                'grid_size': (10, 10),
                'enabled': True
            },
            'hog_features': {
                'func': compute_hog_features,
                'mode': 'gray',
                'target_size': (100, 300),
                'grid_size': (10, 10),
                'enabled': True
            },
            'resnet_avgpool_1024': {
                'extractor_type': 'avgpool_1024',
                'model_name': 'resnet50',
                'enabled': True
            },
            'resnet_layer3_1024': {
                'extractor_type': 'layer3_1024',
                'model_name': 'resnet50',
                'enabled': True
            },
            'resnet_fc_1000': {
                'extractor_type': 'fc_1000',
                'model_name': 'resnet50',
                'enabled': True
            }
        },
        visualize=False,
        output_dir=output_dir
    )


# Example usage
if __name__ == "__main__":
    # Example 1: Extract only ResNet avgpool features
    image_path = "Part1/brain_glioma/brain_glioma_0001.jpg"

    # ResNet features
    #avgpool_features = process_image_resnet(image_path, 'avgpool_1024', 'resnet50')
    #layer3_features = process_image_resnet(image_path, 'layer3_1024', 'resnet50')
    #fc_features = process_image_resnet(image_path, 'fc_1000', 'resnet50')
    #hog_features = process_image_hog(
        #image_path=image_p,
        #visualize=True,
        #output_dir="results"
    #)

    # color_features = process_image_color_moments(
    # image_path=image_p,
    # visualize=True,
    # output_dir="results"
    # )
    # Example 2: Extract all features
    all_features = process_image_all_features(image_path)

    # Example 3: Custom feature combination
    # custom_features = process_image_with_features(
    #     image_path=image_path,
    #     feature_configs={
    #         'color_moments': {
    #             'func': compute_color_moments,
    #             'mode': 'RGB',
    #             'target_size': (100, 300),
    #             'enabled': True
    #         },
    #         'resnet_features': {
    #             'extractor_type': 'avgpool_1024',
    #             'model_name': 'resnet50',
    #             'enabled': True
    #         }
    #     }
    # )