
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import StandardScaler
import traceback
import pickle
import warnings

"""
Task 5 (LS1): Implement a program which (a) given one of the feature models, (b) a user
specified value of k, (c) one of the three dimensionality reduction techniques (SVD, LDA, kmeans) chosen by the user, reports the top-k latent semantics extracted under the selected
feature space.
– Store the latent semantics in a properly named output file
– List imageID-weight pairs, ordered in decreasing order of weights

"""


def load_data_and_label_feature(feature_moment, root_dir = "../Part1"):
    """Load feature vectors and labels from the dataset"""
    feature_vector = feature_moment + ".npy"
    X = []
    y = []
    images_name_collection = []  # Store image IDs for later reference
    for class_label in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_label)
        if not os.path.isdir(class_dir):
            continue
        for subdir, _, files in os.walk(class_dir):
            for file in files:
                if file.endswith(feature_vector):
                    image_name = os.path.basename(subdir)
                    file_path = os.path.join(subdir, file)
                    try:
                        vector = np.load(file_path)
                        if vector.size > 0 and not np.all(vector == 0):
                            X.append(vector)
                            y.append(class_label)
                            # Extract image ID from file path
                            images_name_collection.append(image_name)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        continue

    if not X:
        raise ValueError("No valid vectors found!")
    # Preprocessing: standardization
    scaler = StandardScaler()
    return scaler.fit_transform(np.vstack(X).astype(np.float64)), np.array(y), images_name_collection


def svd_latent_semantics(data,k):
    """
    Extract top-k latent semantics using SVD
    Returns: list of dictionaries with component info and image-weight pairs
    """

    print("Performing SVD...")

    # SVD
    U, s, Vt = np.linalg.svd(data, full_matrices=False)

    # Project data onto first k components
    data_projected = data @ Vt[:k].T
    print(f"New data shape: {data_projected.shape}")
    data_reconstructed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

    return data_projected, data_reconstructed, Vt


def lda_latent_semantics(data, labels, k):
    """
    Extract top-k latent semantics using LDA
    Returns: list of dictionaries with component info and image-weight pairs
    """
    # Maximum number of LDA components = min(n_features, n_classes-1)
    unique_labels = np.unique(labels)
    max_components = min(data.shape[1], len(unique_labels) - 1)
    if max_components < k:
        k = max_components
        print(f" using max_components = {max_components} because {k} is too big")
    print(f"Data shape: {data.shape}")
    print(f"Classes: {len(unique_labels)} → Max LDA components: {k}")

    # LDA
    print("Performing LDA...")
    lda = LinearDiscriminantAnalysis(n_components=k)
    data_projected = lda.fit_transform(data, labels)

    return data_projected, lda


def kmeans_latent_semantics(data, k):
    """
    Extract top-k latent semantics using K-means clustering
    Returns: list of dictionaries with cluster info and image-weight pairs
    """
    km = KMeans(n_clusters= k, random_state=42, n_init='auto')
    km.fit(data)
    centroids = km.cluster_centers_
    # Fase 2: Calcola la distanza di ogni punto dai k centroidi
    data_projected = pairwise_distances(data, centroids, metric='euclidean')
    return data_projected,km


def print_top_k_semantics(data_projected, labels, technique, k, image_name, **kwargs):

    if technique == "kmeans":
        km = kwargs["km"]
        km_labels = km.labels_
        # Calcolo distanze
        distances_to_own_centroid = []
        for i in range(len(km_labels)):
            cluster_label = km_labels[i]
            distance = data_projected[i, cluster_label]  # Distanza già calcolata
            distances_to_own_centroid.append(distance)

        distances_to_own_centroid = np.array(distances_to_own_centroid)
        # Ordinamento completo
        most_representative_indices = np.argsort(distances_to_own_centroid)
        # Ordina tutto insieme
        image_name_ordered = np.array(image_name)[most_representative_indices]
        labels_ordered = labels[most_representative_indices]
        km_labels = np.array(km_labels)[most_representative_indices]
        distances_ordered = distances_to_own_centroid[most_representative_indices]
        # Output
        print(f"Points più rappresentativi:")
        for i in range(0,10):
            print(f"{i + 1}. {image_name_ordered[i]} | Label: {labels_ordered[i]} | Distanza: {distances_ordered[i]:.6f} | cluster num: {km_labels[i]}")
        print("-" * 50)
    elif technique == "svd":
        data_original = kwargs["data_original"]
        data_reconstructed = kwargs["data_reconstructed"]
        reconstruction_errors = np.linalg.norm(data_original - data_reconstructed, axis=1)

        # Ordinamento per rappresentatività
        most_representative_indices = np.argsort(reconstruction_errors)

        print(f"TOP Most Representative Images (SVD)")
        print("=" * 60)
        for i in range(0, 10):
            idx = most_representative_indices[i]
            print(f"Rank {i + 1:2d} | Original Index: {idx:4d} | "
                  f"Image: {image_name[idx]} | "
                  f"Error: {reconstruction_errors[idx]:.6f}")

        print(f"\nDimensions: {data_original.shape} → {data_projected.shape}")
        print(f"Compression ratio: {data_original.shape[1] / data_projected.shape[1]:.1f}x")
        print("-" * 50)
    else:
        unique_labels = np.unique(labels)
        class_centroids = {}
        for label in unique_labels:
            mask = labels == label
            centroid = np.mean(data_projected[mask], axis=0)
            class_centroids[label] = centroid

        # 4. Calcola distanze dai centroidi di classe
        distances_to_class_centroid = []
        for i, point in enumerate(data_projected):
            class_label = labels[i]
            centroid = class_centroids[class_label]
            distance = np.linalg.norm(point - centroid)
            distances_to_class_centroid.append(distance)

        most_representative_indices = np.argsort(distances_to_class_centroid)

        print("TOP K Most Representative Images (LDA)")
        print("=" * 60)

        for i in range(0, 10):
            idx = most_representative_indices[i]
            actual = labels[idx]
            distance_to_own_centroid = distances_to_class_centroid[idx]
            print(f"Rank {i + 1} | Image name: {images_names[idx]} | Label: {actual} | "f"Distance to centroid: {distance_to_own_centroid:.4f}")


def save_latent_semantics(data_projected, labels, technique, k, images_name, feature_type,output_path= "results"):

    # Verifica consistenza lunghezze
    if not (len(labels) == len(images_name) == data_projected.shape[0]):
        raise ValueError("Lunghezze di immagini, etichette e righe di data_projected non corrispondono")

    # Costruisci la lista dei dizionari
    latent_semantics = []
    for i in range(len(images_name)):
        latent_semantics.append({
            "image_id": images_name[i],
            "weights": data_projected[i],  # solo la prima componente (k=1)
            "label": labels[i]
        })

    # Costruisci il dizionario finale
    output_data = {
        "feature_type": feature_type,
        "method": technique,
        "k": k,
        "latent_semantics": latent_semantics
    }
    os.makedirs(output_path, exist_ok=True)
    # Salva su file
    with open(output_path +"/"+str(feature_type) + "_"+str(technique)+ '_'+ str(k)+'.pkl', 'wb') as f:
        pickle.dump(output_data,f)


if __name__ == '__main__':
    # User inputs
    print("Latent Semantics Extraction Tool")
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    # Feature model selection
    feature_models = ["hog_features", "cm10x10_features", "resnet_layer3_1024_features","resnet_avgpool_1024_features", "resnet_fc_1000_features" ]
    print("Available feature models:")
    for i, model in enumerate(feature_models, 1):
        print(f"{i}. {model}")
    choice = int(input("Select feature model (1-5): ")) - 1
    feature_model = feature_models[choice]

    techniques = ["svd", "lda", "kmeans"]
    print("\nAvailable dimensionality reduction techniques:")
    for i, technique in enumerate(techniques, 1):
        print(f"{i}. {technique.upper()}")
    choice = int(input("Select technique (1-3): ")) - 1
    if techniques[choice] != "lda":
        if feature_model == "hog_features":
            k = int(input("Enter the value of k (1<= k <= 900) number of latent semantics : "))
        elif feature_model == "cm10x10_features":
            k = int(input("Enter the value of k (1<= k <= 900): "))
        elif feature_model == "resnet_layer3_1024_features":
            k = int(input("Enter the value of k (1<= k <= 1024): "))
        elif feature_model == "resnet_avgpool_1024_features":
            k = int(input("Enter the value of k (1<= k <= 1024): "))
        elif feature_model == "resnet_fc_1000_features":
            k = int(input("Enter the value of k (1<= k <= 1000): "))
    else:
        k = int(input("Enter the value of k (1<= k <= 2): "))

    try:
        # Load data
        print(f"\nLoading {feature_model} data...")
        data, labels, images_names = load_data_and_label_feature(feature_model)
        print(f"Data loaded: {data.shape}")
        print(f"Number of images: {len(images_names)}")
        print(f"Number of classes: {len(np.unique(labels))}")

        # Extract latent semantics based on selected technique
        if techniques[choice] == "svd":
            data_reduced,data_reconstructed,_ = svd_latent_semantics(data,k)
            kwargs = {"data_original": data, "data_reconstructed": data_reconstructed}
        elif techniques[choice] == "lda":
            data_reduced, lda = lda_latent_semantics(data, labels,k)
            kwargs = {"lda": lda}
        elif techniques[choice] == "kmeans":
            data_reduced, km = kmeans_latent_semantics(data,k)
            kwargs = {"km": km}
        print_top_k_semantics(data_reduced, labels, techniques[choice], k, images_names, **kwargs)
        save_latent_semantics(data_reduced, labels, techniques[choice], k, images_names, feature_model)
    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()
