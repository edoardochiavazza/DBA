import os
import warnings
import numpy as np
import torch
from skdim.id import MLE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torchvision.io import read_image

from Task5 import task5


def load_images(folder_path, target_size=(224, 224)):
    images = []
    labels = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename == ".DS_Store" and os.path.abspath(dirpath) == os.path.abspath(folder_path):
                continue
            # Percorso immagine completo
            image_path = os.path.join(dirpath, filename)
            # Ricava il nome della sottocartella relativa a root_folder_image
            # es: se dirpath = Part1/sottocartella1
            # allora voglio "sottocartella1"
            relative_dir = os.path.relpath(dirpath, folder_path)

            # Carica con PyTorch read_image (formato: C x H x W)
            img = read_image(str(image_path))
            img = img.float()
            # Converti in grayscale se RGB
            if img.shape[0] == 3:  # RGB
                img = torch.mean(img, dim=0, keepdim=True)  # Media dei canali RGB

            # Resize (usando interpolation)
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0).float(),
                size=target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            # Rimuovi dimensione canale se grayscale
            if img.shape[0] == 1:
                img = img.squeeze(0)

            images.append(img.numpy())
            labels.append(relative_dir)
    # Converti in array e flatten
    images = np.array(images, dtype=np.float32)
    print(f"Shape: {images.shape}")

    # Flatten: (n_images, height, width) -> (n_images, height*width)
    data = images.reshape(images.shape[0], -1)

    # Standardizza (consigliato per PCA/SVD)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    print(f"Dati pronti: {data.shape}")

    return data, labels
"""
def load_images_divide_for_labels(folder_path, target_size=(224, 224)):
    vec_images_tumor = []
    vec_images_menin = []
    vec_images_glioma = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename == ".DS_Store" and os.path.abspath(dirpath) == os.path.abspath(folder_path):
                continue
            # Percorso immagine completo
            image_path = os.path.join(dirpath, filename)
            # Ricava il nome della sottocartella relativa a root_folder_image
            # es: se dirpath = Part1/sottocartella1
            # allora voglio "sottocartella1"
            relative_dir = os.path.relpath(dirpath, folder_path)

            # Carica con PyTorch read_image (formato: C x H x W)
            img = read_image(str(image_path))
            img = img.float()
            # Converti in grayscale se RGB
            if img.shape[0] == 3:  # RGB
                img = torch.mean(img, dim=0, keepdim=True)  # Media dei canali RGB

            # Resize (usando interpolation)
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0).float(),
                size=target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            # Rimuovi dimensione canale se grayscale
            if img.shape[0] == 1:
                img = img.squeeze(0)

            if relative_dir == "brain_menin":
                vec_images_menin.append(img)
            elif relative_dir == "brain_glioma":
                vec_images_glioma.append(img)
            else:
                vec_images_tumor.append(img)
    # Converti in array e flatten
    vec_images_tumor = np.array(vec_images_tumor, dtype=np.float32)
    vec_images_menin = np.array(vec_images_menin, dtype=np.float32)
    vec_images_glioma = np.array(vec_images_glioma, dtype=np.float32)

    print(f"Images tumor Shape: {vec_images_tumor.shape}")
    print(f"Images menin Shape: {vec_images_menin.shape}")
    print(f"Images glioma Shape: {vec_images_glioma.shape}")

    # Flatten: (n_images, height, width) -> (n_images, height*width)
    vec_images_tumor = vec_images_tumor.reshape(vec_images_tumor.shape[0], -1)
    vec_images_menin = vec_images_menin.reshape(vec_images_menin.shape[0], -1)
    vec_images_glioma =vec_images_glioma.reshape(vec_images_glioma.shape[0], -1)
    print(f"Data ready for Standar Scaler")
    # Standardizza (consigliato per PCA/SVD)
    scaler = StandardScaler()
    vec_images_tumor = scaler.fit_transform(vec_images_tumor)
    vec_images_menin = scaler.fit_transform(vec_images_menin)
    vec_images_tumor = scaler.fit_transform(vec_images_tumor)

    print(f"Data ready")

    return vec_images_tumor, vec_images_menin, vec_images_glioma
"""
def print_inherith_dimension_svd(data):
    """
    Extract top-k latent semantics using SVD
    Returns: list of dictionaries with component info and image-weight pairs
    """
    print("Performing SVD...")

    # SVD
    U, s, Vt = np.linalg.svd(data, full_matrices=False)

    # Calculate explained variance
    explained_variance = s ** 2 / (len(data) - 1)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    # Cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Select number of components to preserve 95% variance
    best_num = np.argmax(cumulative_variance >= 0.90) + 1
    print(f"Best number of components for maintain 90% variance: {best_num} applying SVD")
    best_num_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Best number of components for maintain 95% variance: {best_num_95} applying SVD")
    best_num = np.argmax(cumulative_variance >= 0.99) + 1
    print(f"Best number of components for maintain 99% variance: {best_num} applying SVD")

    return best_num_95

def lda_latent_semantics(data, labels):
    print("Performing LDA...")
    lda = LinearDiscriminantAnalysis()
    lda.fit(data, labels)

    # Autovalori delle componenti discriminanti
    eigenvalues = lda.explained_variance_ratio_

    # Percentuale cumulativa
    cumulative = np.cumsum(eigenvalues)

    # Select number of components to preserve 95% variance
    best_num = np.argmax(cumulative >= 0.90) + 1
    print(f"Best number of components for maintain 90% variance applying LDA: {best_num}")
    best_num_95 = np.argmax(cumulative >= 0.95) + 1
    print(f"Best number of components for maintain 95% variance applying LDA: {best_num_95}")
    best_num = np.argmax(cumulative >= 0.99) + 1
    print(f"Best number of components for maintain 99% variance applying LDA: {best_num}")

    return best_num_95

def pca_latent_semantics(data):
    print("Performing PCA...")
    pca = PCA()
    pca.fit(data)
    # Calcolo varianza cumulativa
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    # Select number of components to preserve 95% variance
    best_num = np.argmax(cum_var >= 0.90) + 1
    print(f"Best number of components for maintain 90% variance applying PCA: {best_num}")
    best_num_95 = np.argmax(cum_var >= 0.95) + 1
    print(f"Best number of components for maintain 95% variance applying PCA: {best_num_95}")
    best_num = np.argmax(cum_var >= 0.99) + 1
    print(f"Best number of components for maintain 99% variance applying PCA: {best_num}")

    return best_num_95

def kmeans_latent_semantics(data):
    """
    Extract top-k latent semantics using K-means clustering
    Returns: list of dictionaries with cluster info and image-weight pairs
    """
    # 1. Trova il k migliore
    best_k = None
    best_score = -1

    for i in range(2, 10):
        km = KMeans(n_clusters=i, random_state=42, n_init='auto')
        labels = km.fit_predict(data)
        score = silhouette_score(data, labels)
        if score > best_score:
            best_score = score
            best_k = i

    print(f"Il miglior numero di cluster è {best_k} con silhouette score = {best_score:.4f}")

def mle_latent_semantics(data):
    print("Performing MLE...")
    est = MLE()
    best_dim = est.fit(data).dimension_
    print(f"MLE estimated dimension: {best_dim}")
    return round(best_dim, 0)

if __name__ == '__main__':

    warnings.filterwarnings('ignore', category=RuntimeWarning)
    path = "../Part1"
    features_vect = ["hog_features", "cm10x10_features", "resnet_avgpool_1024_features", "resnet_fc_1000_features",
                     "resnet_layer3_1024_features", "rgb"]
    for feature in features_vect:

        if feature == "rgb":
            data, labels = load_images(path, target_size=(224, 224))
        else:
            data, labels, _ = task5.load_data_and_label_feature(feature, root_dir="../Task2/new_results")
        data_train = np.array(data)
        labels_train = np.array(labels)
        print("Performing inherent dimensionality associated with the part 1 images")
        print_inherith_dimension_svd(data)
        lda_latent_semantics(data, labels)
        pca_latent_semantics(data)
        mle_latent_semantics(data)
        dict_data_test = {"brain_tumor": data_train[labels_train == "brain_tumor"],
                          "brain_menin": data_train[labels_train == "brain_menin"],
                          "brain_glioma": data_train[labels_train == "brain_glioma"]}
        for key in dict_data_test:
            print(f"Performing inherent dimensionality for {key} label ")
            print_inherith_dimension_svd(dict_data_test[key])
            pca_latent_semantics(dict_data_test[key])
            mle_latent_semantics(dict_data_test[key])

