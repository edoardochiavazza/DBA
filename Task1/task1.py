import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from skimage.feature import hog
from torchvision.io import read_image


class ResNetFeatureExtractor:
    """ResNet-based feature extractor for deep learning features"""

    def __init__(self, model_name='resnet50', pretrained=True):
        """
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


        self.hooked_features = {}
        self.hooked_gradients = {}

    def _forward_hook_fn(self, name):
        """Create forward hook function"""

        def hook(module, input, output):
            self.hooked_features[name] = output.detach()

        return hook

    def _backward_hook_fn(self, name):
        """Create backward hook function"""

        def hook(module, grad_input, grad_output):
            self.hooked_gradients[name] = grad_output[0].detach()

        return hook

    def _preprocess_image(self, image_tensor):

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
        hook = self.model.avgpool.register_forward_hook(self._forward_hook_fn('avgpool'))

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
        hook = self.model.layer3.register_forward_hook(self._forward_hook_fn('layer3'))

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
        hook = self.model.fc.register_forward_hook(self._forward_hook_fn('fc'))

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

    def compute_grad_map(self, image_tensor, layer_name, target_class=None):
        layer = layer_name.split('_')[0]

        # Find the target layer in the model
        target_layer = None
        for name, module in self.model.named_modules():
            if name == layer or name.endswith(layer):
                target_layer = module
                break

        if target_layer is None:
            raise ValueError(f"Layer '{layer}' not found in model")

        # Register both forward and backward hooks
        forward_hook = target_layer.register_forward_hook(self._forward_hook_fn(layer))
        back_hook = target_layer.register_full_backward_hook(self._backward_hook_fn(layer))

        try:
            input_tensor = self._preprocess_image(image_tensor)
            input_tensor.requires_grad_(True)  # Enable gradient computation

            # Forward pass (remove no_grad to allow gradients)
            self.model.eval()
            output = self.model(input_tensor)

            # Determine target class
            if target_class is None:
                target_class = output.argmax(dim=1).item()

            # Backward pass to compute gradients
            self.model.zero_grad()
            output[0, target_class].backward()

            # Get the hooked gradients and activations
            gradients = self.hooked_gradients[layer]
            activations = self.hooked_features[layer]
            # Compute Grad-CAM style weights
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            weighted_activations = torch.sum(weights * activations, dim=1).squeeze()
            activation_map = weighted_activations.cpu().numpy()

            # Normalizzazione
            activation_map = np.maximum(activation_map, 0)  # ReLU
            if activation_map.max() > activation_map.min():
                activation_map = (activation_map - activation_map.min()) / (
                        activation_map.max() - activation_map.min() + 1e-8)
            else:
                activation_map = np.zeros_like(activation_map)

            # Ridimensiona all'originale
            heatmap = cv2.resize(activation_map, (image_tensor.shape[2], image_tensor.shape[1]))
            return heatmap

        finally:
            forward_hook.remove()
            back_hook.remove()
            self.hooked_gradients.clear()
            self.hooked_features.clear()


def load_and_resize_image(image_path, target_size=(100, 300), mode='RGB',use_cropping=False):
    """
    Args:
        image_path (str): Path to the input image
        target_size (tuple): Target size as (height, width)
        mode (str): The desired image format, either 'gray' or 'RGB'
        :param image_path:
        :param target_size:
        :param mode:
        :param use_cropping: choose to crop image

    Returns:
        torch.Tensor: Resized image tensor with shape (C, height, width) and values in [0,1]

    """
    try:
        img = read_image(image_path)
        if use_cropping:
            img = crop_image(img[0, :, :])
            img = np.expand_dims(img, axis=0)
            img = torch.from_numpy(img)
        if img.dtype == torch.uint8: # Passo da uint8 [0,255] a float [0,1]
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
                          visualize=False, output_dir="Task1/results", image_path=None, use_cropping=False):
    grid_rows, grid_cols = grid_size

    if image_name and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    feature_vector = np.zeros((grid_rows, grid_cols, 3, 3))
    cell_idx = 0
    for i in range(grid_rows):
        for j in range(grid_cols):
            cell = grid_cells[cell_idx]
            cell_np = cell.numpy()
            # Aggiungi controlli per identificare la fonte:
            for channel in range(3):  # RGB channels
                # Mean (1st moment)
                mean = np.mean(cell_np[channel])
                feature_vector[i, j, channel, 0] = mean
                # Standard deviation (2nd moment)
                std = np.std(cell_np[channel])
                feature_vector[i, j, channel, 1] = std
                # Skewness (3rd moment) L’asimmetria distribuzione è sbilanciata rispetto alla sua media.
                channel_data = cell_np[channel].flatten()
                # Rimuovi NaN e controlla se rimangono abbastanza valori
                valid_data = channel_data[~np.isnan(channel_data)]

                if len(valid_data) < 3 or np.std(valid_data) < 1e-10:
                    skewness = 0
                else:
                    skewness = stats.skew(valid_data)
                    if np.isnan(skewness):
                        skewness = 0
                feature_vector[i, j, channel, 2] = skewness
            cell_idx += 1

    if visualize and image_name:
        img_resized = reconstruct_image_from_cells(grid_cells, grid_size)
        visualize_color_moments(img_resized, feature_vector, output_dir)

    flattened_features = feature_vector.flatten()

    if image_name and output_dir:
        np.save(os.path.join(output_dir, "cm10x10_features.npy"), flattened_features)

    return flattened_features

def visualize_hog_features(hog_features, image_path, output_dir,use_cropping=False):

    # Estrai magnitudo media per cella
    hog_matrix = hog_features.reshape(10, 10, 9)
    magnitude_map = hog_matrix.mean(axis=2)
    img = read_image(image_path)
    if use_cropping:
        img = crop_image(img[0, :, :])
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.imshow(img[0], cmap='gray')
    plt.title("Originale")

    plt.subplot(122)
    plt.imshow(magnitude_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Magnitudo media gradienti')
    plt.title("Map HOG (10x10)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Map_red_hog.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Con cell_height=30, cell_width=10, griglia 10x10
    row_labels = [f"Py_{i*30}-{(i+1)*30}" for i in range(10)]  # Y_0-30, Y_30-60, etc.
    col_labels = [f"Px_{j*10}-{(j+1)*10}" for j in range(10)]  # X_0-10, X_10-20, etc.
    annotations = np.where(magnitude_map >= 0.4,
                           np.round(magnitude_map, 2).astype(str),
                           '')
    sns.heatmap(magnitude_map, cmap='gray',
                xticklabels=col_labels,
                yticklabels=row_labels,
                annot=annotations, fmt='', cbar_kws={'label': 'Intensità del gradiente'})

    plt.savefig(os.path.join(output_dir,'heatmapHOG.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # Estrai HOG con i tuoi parametri (10x10 grid, 9 bin signed)
    _, hog_image = hog(
        img[0,:,:],
        pixels_per_cell=(30, 10),  # 300x100 / 10x10 = 30x10 per cella
        cells_per_block=(1, 1),
        orientations=9,
        visualize=True,
        feature_vector=True
    )

    plt.figure(figsize=(15, 5))
    # Originale + HOG overlay
    plt.imshow(img[0], cmap='gray')
    plt.imshow(hog_image, alpha=0.3, cmap='jet')  # Gradienti HOG standard
    plt.title("Original + HOG")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Originale+HOG.png"), dpi=300, bbox_inches='tight')
    plt.close()

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


def compute_hog_features(grid_cells, image_name=None,
                         output_dir="Task1/results", visualize=True, **kwargs):
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
    if visualize:
        visualize_hog_features(features_descriptor, image_path, output_dir)
    if image_name and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "hog_features.npy"), features_descriptor)

    return features_descriptor


def compute_resnet_features(img_tensor, extractor_type='avgpool_1024', image_name=None,
                            output_dir="Task1/results", model_name='resnet50', visualize = False,**kwargs):
    """
    Compute ResNet features using the specified extraction method.

    Args:
        img_tensor (torch.Tensor): Input image tensor
        extractor_type (str): Type of extraction ('avgpool_1024', 'layer3_1024', 'fc_1000')
        image_name (str): Name for saving features
        output_dir (str): Output directory
        model_name (str): ResNet model to use
         :param visualize:

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
        feature_filename = f"resnet_{extractor_type}_features.npy"
        np.save(os.path.join(output_dir, feature_filename), features)
    if visualize and extractor_type == 'layer3_1024' and output_dir is not None:
        act_map= extractor.compute_grad_map(img_tensor,extractor_type)
        # Visualizzazione
        plt.imshow(act_map, cmap='jet')
        plt.colorbar()
        plt.title("Activation map for layer3")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "Activation_map_for_layer3.png"), dpi=300, bbox_inches='tight')
        plt.close()


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


def visualize_grid_cells(img_tensor, output_dir="Task1/results"):
    """Create visualizations of the grid and color moments."""

    # Converti il tensor PyTorch in numpy array per la visualizzazione
    if isinstance(img_tensor, torch.Tensor):
        # Se è un tensor con shape (C, H, W)
        if len(img_tensor.shape) == 3:
            if img_tensor.shape[0] == 1:  # Scala di grigi
                img_np = img_tensor[0].numpy()  # Prendi solo il primo canale
                cmap = 'gray'
            elif img_tensor.shape[0] == 3:  # RGB
                img_np = img_tensor.permute(1, 2, 0).numpy()  # Converti in (H, W, C)
                cmap = None
            else:
                # Fallback per altri formati
                img_np = img_tensor[0].numpy()
                cmap = 'gray'
        else:
            # Se è già in formato (H, W)
            img_np = img_tensor.numpy()
            cmap = 'gray'
    else:
        # Se è già un numpy array
        img_np = img_tensor
        cmap = 'gray' if len(img_np.shape) == 2 else None

    # Assicurati che i valori siano nel range corretto [0, 1]
    img_np = np.clip(img_np, 0.0, 1.0)

    # Grid visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.imshow(img_np, cmap=cmap)  # Usa la colormap appropriata

    grid_rows, grid_cols = 10, 10

    # Ottieni le dimensioni corrette dell'immagine
    if len(img_np.shape) == 3:  # RGB
        height, width = img_np.shape[:2]
    else:  # Scala di grigi
        height, width = img_np.shape

    cell_height = height // grid_rows
    cell_width = width // grid_cols

    # Disegna le linee della griglia
    for i in range(grid_rows + 1):
        ax.axhline(i * cell_height, color='red', linestyle='-', linewidth=0.5)
    for j in range(grid_cols + 1):
        ax.axvline(j * cell_width, color='red', linestyle='-', linewidth=0.5)

    ax.set_title(f"{width}×{height} Image with {grid_cols}×{grid_rows} Grid")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    plt.tight_layout()

    if output_dir is None:
        return  # oppure salta la visualizzazione

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "grid_visualization.png"), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_color_moments(img_tensor, feature_vector, output_dir):
    """Create visualizations of the grid and color moments."""
    img_np = img_tensor.permute(1, 2, 0).numpy()
    np.clip(img_np, 0.0, 1.0)

    grid_rows, grid_cols = 10, 10
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


def process_image_with_features(image_path, feature_configs, visualize=False, output_dir="Task1/results", use_cropping=False):
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
    if output_dir is  not None:
        output_dir = os.path.join(output_dir, image_name)
        os.makedirs(output_dir, exist_ok=True)
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
            img_tensor = load_and_resize_image(image_path, target_size=(224, 224), mode='RGB', use_cropping=False)
            features = compute_resnet_features(
                img_tensor=img_tensor,
                extractor_type=config['extractor_type'],
                image_name=image_name,
                output_dir=output_dir,
                model_name=config.get('model_name', 'resnet50'),
                use_cropping = use_cropping
            )
        else:
            # Grid-based features
            img_tensor = load_and_resize_image(image_path, target_size=target_size, mode=mode)
            visualize_grid_cells(img_tensor,output_dir)
            grid_cells, _ = compute_grid_cells(img_tensor, grid_size=config.get('grid_size', (10, 10)))
            func = config['func']
            params = config.get('params', {})
            params.update({
                'image_name': image_name,
                'output_dir': output_dir,
                'visualize': visualize,
                'image_path': image_path,
                'use_cropping': use_cropping
            })

            features = func(grid_cells, **params)

        results[feature_name] = features
        analyze_feature_vector(features, feature_name)

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

def crop_image(img):
    img = img.numpy()
    print(f"Image type: {img.dtype}")
    print(f"Image shape: {img.shape}")

    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        if img.dtype == np.float32:
            # Assuming values are in range [0, 1], scale to [0, 255]
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # Find non-black edges (thresholding)
    _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if contours were found
    if len(contours) == 0:
        print("No contours found! Image might be completely black or empty.")
        print(f"Image min: {img.min()}, max: {img.max()}")
        print(f"Thresh min: {thresh.min()}, max: {thresh.max()}")
        # Return original image or handle error as needed
        return img

    # Get the largest contour (assuming it's the main object)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the ROI
    cropped_img = img[y:y + h, x:x + w]

    cv2.imwrite('cropped_img.jpg', cropped_img)
    return cropped_img



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


def process_image_all_features(image_path, output_dir="Task1/results", visualize=False, use_cropping = False):
    """Extract all types of features from an image."""
    return process_image_with_features(
        image_path=image_path,
        feature_configs={
            'color_moments': {
                'func': compute_color_moments,
                'mode': 'RGB',
                'target_size': (100, 300),
                'grid_size': (10, 10),
                'enabled': True,
                'image_path': image_path,
            },
            'hog_features': {
                'func': compute_hog_features,
                'mode': 'gray',
                'target_size': (100, 300),
                'grid_size': (10, 10),
                'enabled': True,
                'image_path': image_path,
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
        visualize=visualize,
        use_cropping=use_cropping,
        output_dir=output_dir
    )


# Example usage
if __name__ == "__main__":
    # Example 1: Extract only ResNet avgpool features
    image_path = "../Part1/brain_menin/brain_menin_0108.jpg"

    # ResNet features
    #avgpool_features = process_image_resnet(image_path, 'avgpool_1024', 'resnet50')
    #layer3_features = process_image_resnet(image_path, 'layer3_1024', 'resnet50')
    #fc_features = process_image_resnet(image_path, 'fc_1000', 'resnet50')
    #hog_features = process_image_hog(
        #image_path=image_p,
        #visualize=True,
        #output_dir="results"
    #)
    """
    color_features = process_image_color_moments(
    image_path=image_path,
    visualize=False, 
    output_dir=None)
    print(f"\n cm = {color_features.shape}")
    """
    # Example 2: Extract all features
    all_features = process_image_all_features(image_path, visualize=True, use_cropping = True, output_dir="")

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