import numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors

from Task5 import task5
from Task6 import task6
from Task6.task6 import pca_latent_semantics, load_images

"""
implement a program which,
– for each unique label l, computes the correspending c most significant clusters associated
with the part 1 images (using DBScan algorithm); the resulting clusters should be visualized
both
∗ as differently colored point clouds in a 2-dimensional MDS space, and
∗ as groups of image thumbnails.
"""


# Funzione helper per caricare immagini se necessario
def load_images_from_paths(image_paths, label, target_size=(64, 64)):
    """
    Carica immagini da una lista di percorsi

    Args:
        image_paths: lista di percorsi alle immagini
        target_size: dimensione target per le immagini

    Returns:
        array di immagini
    """
    full = "../Part1/"+label+"/"
    from PIL import Image
    images = []
    for path in image_paths:
        try:
            img = Image.open(full+path+".jpg")
            img = img.resize(target_size)
            img_array = np.array(img)
            images.append(img_array)
        except Exception as e:
            print(f"Errore caricando {path}: {e}")
            # Crea immagine placeholder
            placeholder = np.zeros((*target_size, 3), dtype=np.uint8)
            images.append(placeholder)

    return np.array(images)

def plot_all_datasets_together(dict_data_test, centroids_dict_dbscan, dict_sorted_cluster, c=3, title=f'Plot Tutti i Dataset di ogni label'):
    """
    Plotta i top c cluster di tutti i dataset insieme con colori diversi
    Ogni dataset ha colori diversi, incluso il rumore
    """
    plt.figure(figsize=(15, 10))

    # Colori base per i dataset
    dataset_colors = plt.cm.Set1(np.linspace(0, 1, len(dict_data_test)))

    for dataset_idx, key in enumerate(dict_data_test.keys()):
        data = dict_data_test[key]
        labels = centroids_dict_dbscan[key]
        sorted_keys = dict_sorted_cluster[key]

        # Riduzione dimensionalità se necessario
        if data.shape[1] > 2:
            data = PCA(n_components=2).fit_transform(data)

        # Determina quali cluster plottare (top c)
        unique_labels = set(labels)
        sorted_list = list(sorted_keys)[:c] if c <= len(sorted_keys) else list(sorted_keys)
        clusters_to_plot = set(sorted_list)

        # Aggiungi sempre il rumore se presente
        if -1 in unique_labels:
            clusters_to_plot.add(-1)

        # Genera colori per questo dataset
        base_color = dataset_colors[dataset_idx]
        n_clusters_to_plot = len(clusters_to_plot)

        # Crea variazioni di colore per i cluster di questo dataset
        cluster_colors = []
        for i in range(n_clusters_to_plot):
            # Varia la luminosità del colore base
            brightness = 0.3 + 0.7 * i / max(1, n_clusters_to_plot - 1)
            color = [base_color[0] * brightness, base_color[1] * brightness,
                     base_color[2] * brightness, base_color[3]]
            cluster_colors.append(color)

        # Plotta ogni cluster
        for cluster_idx, label in enumerate(sorted(clusters_to_plot)):
            mask = labels == label
            points = data[mask]

            if len(points) == 0:
                continue

            color = cluster_colors[cluster_idx % len(cluster_colors)]

            if label == -1:  # Rumore
                plt.scatter(points[:, 0], points[:, 1], c=[color], marker='x',
                            s=30, alpha=0.7, label=f'{key} - Rumore')
            else:
                plt.scatter(points[:, 0], points[:, 1], c=[color], marker='o',
                            s=50, alpha=0.7, label=f'{key} - Cluster {label}')

    plt.title(title)
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_clusters(data, labels, title="Clusters", exclude=None, c=None, sorted_keys=None):
    """
    Plotta cluster con colori diversi

    Args:
        data: dati da plottare
        labels: etichette cluster
        title: titolo del grafico
        exclude: lista cluster da escludere (es. [0,1] o [-1] per rumore)
        c: numero primi cluster da plottare (usa sorted_keys)
        sorted_keys: cluster ordinati per qualità
    """
    # Riduzione dimensionalità se necessario
    if data.shape[1] > 2:
        data = PCA(n_components=2).fit_transform(data)

    # Determina quali cluster plottare
    unique_labels = set(labels)

    if c is not None and sorted_keys is not None:
        # Plotta solo i primi c cluster
        sorted_list = list(sorted_keys)[:c] if c <= len(sorted_keys) else list(sorted_keys)
        exclude = list(unique_labels - set(sorted_list) - {-1})  # Mantieni rumore

    if exclude:
        unique_labels = unique_labels - set(exclude)

    # Plot
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        points = data[mask]

        if label == -1:  # Rumore
            plt.scatter(points[:, 0], points[:, 1], c='black', marker='x',
                        s=20, alpha=0.7, label='Rumore')
        else:
            plt.scatter(points[:, 0], points[:, 1], c=[color], marker='o',
                        s=50, alpha=0.7, label=f'Cluster {label}')

    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_clusters_with_images(data, labels, images, title="Clusters", exclude=None, c=None, sorted_keys=None,
                          thumbnail_size=50, max_images_per_cluster=20):
    """
    Plotta cluster come gruppi di miniature di immagini

    Args:
        data: dati da plottare (features)
        labels: etichette cluster
        images: array di immagini corrispondenti ai dati
        title: titolo del grafico
        exclude: lista cluster da escludere
        c: numero primi cluster da plottare
        sorted_keys: cluster ordinati per qualità
        thumbnail_size: dimensione thumbnail in pixel
        max_images_per_cluster: massimo numero di immagini per cluster da mostrare
    """
    # Riduzione dimensionalità se necessario
    if data.shape[1] > 2:
        data_2d = PCA(n_components=2).fit_transform(data)
    else:
        data_2d = data

    # Determina quali cluster plottare
    unique_labels = set(labels)

    if c is not None and sorted_keys is not None:
        sorted_list = list(sorted_keys)[:c] if c <= len(sorted_keys) else list(sorted_keys)
        exclude = list(unique_labels - set(sorted_list) - {-1})  # Mantieni rumore

    if exclude:
        unique_labels = unique_labels - set(exclude)

    # Crea figura
    fig, ax = plt.subplots(figsize=(15, 12))

    # Colori per i cluster
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        cluster_data = data_2d[mask]
        cluster_images = images[mask]

        if len(cluster_data) == 0:
            continue

        # Calcola il centroide del cluster
        centroid = np.mean(cluster_data, axis=0)

        # Limita il numero di immagini da mostrare
        if len(cluster_images) > max_images_per_cluster:
            # Seleziona immagini casuali o le prime N
            indices = np.random.choice(len(cluster_images), max_images_per_cluster, replace=False)
            cluster_images = cluster_images[indices]
            cluster_data = cluster_data[indices]

        # Arrange images in a grid around centroid
        n_images = len(cluster_images)
        grid_size = int(np.ceil(np.sqrt(n_images)))

        # Calcola offset per posizionare le immagini attorno al centroide
        spacing = thumbnail_size * 0.8 / 1000  # Converti in coordinate del grafico

        for i, (img, point) in enumerate(zip(cluster_images, cluster_data)):
            # Posizione nell'array di immagini
            row = i // grid_size
            col = i % grid_size

            # Calcola offset dalla posizione originale del punto
            offset_x = (col - grid_size // 2) * spacing * 0.5
            offset_y = (row - grid_size // 2) * spacing * 0.5

            # Posizione finale
            x = point[0] + offset_x
            y = point[1] + offset_y

            # Normalizza l'immagine se necessario
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            # Crea OffsetImage
            imagebox = OffsetImage(img, zoom=thumbnail_size / 100)

            # Aggiungi bordo colorato per indicare il cluster
            if label == -1:  # Rumore
                imagebox.image.axes = ax
                ab = AnnotationBbox(imagebox, (x, y), frameon=True,
                                    boxcoords="data", pad=0.1)
                ab.patch.set_edgecolor('red')
                ab.patch.set_linewidth(2)
            else:
                imagebox.image.axes = ax
                ab = AnnotationBbox(imagebox, (x, y), frameon=True,
                                    boxcoords="data", pad=0.1)
                ab.patch.set_edgecolor(color)
                ab.patch.set_linewidth(2)

            ax.add_artist(ab)

        # Aggiungi etichetta del cluster al centroide
        if label == -1:
            ax.text(centroid[0], centroid[1], 'Rumore',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7),
                    fontsize=12, ha='center', va='center', weight='bold')
        else:
            ax.text(centroid[0], centroid[1], f'Cluster {label}',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                    fontsize=12, ha='center', va='center', weight='bold')

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Componente 1', fontsize=12)
    ax.set_ylabel('Componente 2', fontsize=12)
    ax.grid(alpha=0.3)

    # Aggiusta i limiti degli assi per includere tutte le immagini
    margin = 0.1
    ax.set_xlim(data_2d[:, 0].min() - margin, data_2d[:, 0].max() + margin)
    ax.set_ylim(data_2d[:, 1].min() - margin, data_2d[:, 1].max() + margin)

    plt.tight_layout()
    plt.show()


def plot_all_datasets_with_images(dict_data_test, centroids_dict_dbscan, dict_images,
                                  dict_sorted_cluster, c=3, thumbnail_size=0.15,
                                  max_images_per_cluster=10,
                                  title='Plot Tutti i Dataset con Immagini'):
    """
    Plotta i top c cluster di tutti i dataset insieme usando miniature delle immagini

    Args:
        dict_data_test: dizionario con i dati di test
        dict_images: dizionario con le immagini corrispondenti ai dati
        centroids_dict_dbscan: dizionario con le etichette dei cluster
        dict_sorted_cluster: dizionario con i cluster ordinati
        c: numero di cluster top da plottare
        thumbnail_size: dimensione thumbnail (come frazione della figura)
        max_images_per_cluster: massimo numero di immagini per cluster
        title: titolo del grafico
    """
    fig, ax = plt.subplots(figsize=(20, 15))

    # Colori base per i dataset
    dataset_colors = plt.cm.Set1(np.linspace(0, 1, len(dict_data_test)))

    # Offset per separare i dataset visivamente
    dataset_offset = 0
    max_extent = 0

    for dataset_idx, key in enumerate(dict_data_test.keys()):
        data = dict_data_test[key]
        images = dict_images[key]
        labels = centroids_dict_dbscan[key]
        sorted_keys = dict_sorted_cluster[key]

        # Riduzione dimensionalità se necessario
        if data.shape[1] > 2:
            data_2d = PCA(n_components=2).fit_transform(data)
        else:
            data_2d = data.copy()

        # Sposta il dataset per evitare sovrapposizioni
        data_2d[:, 0] += dataset_offset
        dataset_extent = data_2d[:, 0].max() - data_2d[:, 0].min()
        max_extent = max(max_extent, dataset_extent)
        dataset_offset += dataset_extent + 3

        # Determina quali cluster plottare
        unique_labels = set(labels)
        if len(sorted_keys) > 0:
            sorted_list = list(sorted_keys)[:c] if c <= len(sorted_keys) else list(sorted_keys)
            clusters_to_plot = set(sorted_list)
        else:
            clusters_to_plot = unique_labels

        # Aggiungi sempre il rumore se presente
        if -1 in unique_labels:
            clusters_to_plot.add(-1)

        # Colore base per questo dataset
        base_color = dataset_colors[dataset_idx]

        for label in clusters_to_plot:
            mask = labels == label
            cluster_data = data_2d[mask]
            cluster_images = images[mask]

            if len(cluster_data) == 0:
                continue

            print(f"  Cluster {label}: {len(cluster_data)} samples")

            # Calcola centroide
            centroid = np.mean(cluster_data, axis=0)

            # Limita numero di immagini
            if len(cluster_images) > max_images_per_cluster:
                indices = np.random.choice(len(cluster_images), max_images_per_cluster, replace=False)
                cluster_images = cluster_images[indices]
                cluster_data = cluster_data[indices]

            # Posiziona le immagini in modo più distribuito
            n_images = len(cluster_images)

            # Calcola posizioni in cerchio attorno al centroide
            angles = np.linspace(0, 2 * np.pi, n_images, endpoint=False)
            radius = 0.5  # Raggio del cerchio

            for i, (img, angle) in enumerate(zip(cluster_images, angles)):
                # Posizione in cerchio attorno al centroide
                x = centroid[0] + radius * np.cos(angle)
                y = centroid[1] + radius * np.sin(angle)

                # Gestisci il tipo di immagine
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)

                # Assicurati che l'immagine sia nel formato corretto
                if len(img.shape) == 3 and img.shape[2] == 3:
                    display_img = img
                elif len(img.shape) == 2:
                    display_img = np.stack([img, img, img], axis=-1)
                else:
                    print(f"Formato immagine non supportato: {img.shape}")
                    continue

                try:
                    # Crea thumbnail con dimensione fissa
                    imagebox = OffsetImage(display_img, zoom=thumbnail_size, resample=True)

                    # Colore bordo
                    if label == -1:
                        border_color = 'red'
                    else:
                        border_color = base_color

                    ab = AnnotationBbox(imagebox, (x, y), frameon=True,
                                        boxcoords="data", pad=0.02)
                    ab.patch.set_edgecolor(border_color)
                    ab.patch.set_linewidth(2)
                    ab.patch.set_facecolor('white')
                    ax.add_artist(ab)

                except Exception as e:
                    print(f"Errore nell'aggiungere l'immagine: {e}")
                    # Aggiungi un punto come fallback
                    ax.plot(x, y, 'o', color=border_color, markersize=8)

            # Etichetta cluster al centroide
            if label == -1:
                ax.text(centroid[0], centroid[1], f'{key}\nRumore',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.8),
                        fontsize=10, ha='center', va='center', weight='bold',
                        color='white')
            else:
                ax.text(centroid[0], centroid[1], f'{key}\nCluster {label}',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=base_color, alpha=0.8),
                        fontsize=10, ha='center', va='center', weight='bold',
                        color='white')

    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xlabel('Componente 1', fontsize=14)
    ax.set_ylabel('Componente 2', fontsize=14)
    ax.grid(alpha=0.3)

    # Imposta i limiti degli assi con margini adeguati
    # Calcola i limiti da tutti i dataset trasformati
    x_coords = []
    y_coords = []

    current_offset = 0
    for key in dict_data_test.keys():
        data = dict_data_test[key]
        if data.shape[1] > 2:
            data_2d = PCA(n_components=2).fit_transform(data)
        else:
            data_2d = data.copy()

        # Applica l'offset
        data_2d[:, 0] += current_offset
        dataset_extent = data_2d[:, 0].max() - data_2d[:, 0].min()
        current_offset += dataset_extent + 3

        x_coords.extend(data_2d[:, 0])
        y_coords.extend(data_2d[:, 1])

    # Aggiungi margini
    x_min = min(x_coords) - 1
    x_max = max(x_coords) + 1
    y_min = min(y_coords) - 1
    y_max = max(y_coords) + 1

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()

def sorted_cluster(data, labels):
    """
    Trova il cluster con il silhouette score migliore
    """
    # Controlla se ci sono abbastanza cluster
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        print("Non abbastanza cluster per l'analisi")
        return None

    # Calcola silhouette per ogni sample
    sample_silhouette = silhouette_samples(data, labels) # calcola quanto il punto è ben

    # Calcola score medio per ogni cluster
    cluster_scores = {}
    for label in unique_labels:
        cluster_mask = labels == label
        avg_score = np.mean(sample_silhouette[cluster_mask])
        cluster_size = np.sum(cluster_mask)

        cluster_scores[label] = {
            'silhouette': avg_score,
            'size': cluster_size
        }

    # Trova il migliore
    best_cluster = max(cluster_scores.keys(), key=lambda x: cluster_scores[x]['silhouette'])
    best_score = cluster_scores[best_cluster]['silhouette']

    # Stampa risultati
    print(f"  Silhouette globale: {silhouette_score(data, labels):.3f}")
    for label in unique_labels:
        marker = "Best:" if label == best_cluster else "     "
        print(
            f"  {marker} Cluster {label}: {cluster_scores[label]['silhouette']:.3f} (n={cluster_scores[label]['size']})")
    cluster_scores = dict(sorted(cluster_scores.items(), key=lambda x: x[1]['silhouette'], reverse=True))
    return cluster_scores.keys()

def find_optimal_eps(data, k=5):
    """Trova eps ottimale usando k-distance plot"""

    # Calcola distanze ai k vicini più prossimi
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)

    # Ordina le distanze
    distances = np.sort(distances, axis=0)
    distances = distances[:, k - 1]  # k-esima distanza

    # Suggerimento automatico (punto di massima curvatura)
    # Metodo semplice: usa percentile
    suggested_eps = np.percentile(distances, 95)
    print(f"Eps suggerito: {suggested_eps:.4f}")

    return suggested_eps





if __name__ == '__main__':

    features_vect = ["hog_features", "cm10x10_features", "resnet_avgpool_1024_features", "resnet_fc_1000_features", "resnet_layer3_1024_features"]
    #features_vect = ["rgb"]
    path_part1 = "../Part1"
    for feature in features_vect:
        print(f"Using feature space: {feature}")
        if feature == "rgb":
            data_train, labels_train = task6.load_images(path_part1, target_size=(224, 224))
        else:
            data_train, labels_train, images_names_train = task5.load_data_and_label_feature(feature, root_dir="../Task2/new_results")
        data_train = np.array(data_train)
        labels_train = np.array(labels_train)
        images_names_train = np.array(images_names_train)
        dict_image = {"brain_tumor": load_images_from_paths(images_names_train[labels_train == "brain_tumor"], "brain_tumor"),
                          "brain_menin": load_images_from_paths(images_names_train[labels_train == "brain_menin"],"brain_menin"),
                          "brain_glioma": load_images_from_paths(images_names_train[labels_train == "brain_glioma"],"brain_glioma")}
        dict_data_test = {"brain_tumor": data_train[labels_train == "brain_tumor"],
                          "brain_menin": data_train[labels_train == "brain_menin"],
                          "brain_glioma": data_train[labels_train == "brain_glioma"]}
        predictions_test_cosine = []
        predictions_test_euclidean = []
        centroids_dict_dbscan = {}
        best_dict_eps = {}
        # Usa per ogni feature space
        for key in dict_data_test.keys():
            print(f"\nAnalizzando {key}...")
            best_k = pca_latent_semantics(dict_data_test[key])
            print(f"data shape: {dict_data_test[key].shape}")
            pca = PCA(n_components=best_k)
            dict_data_test[key] = pca.fit_transform(dict_data_test[key])
            print(f"data shape after pca: {dict_data_test[key].shape}")
            best_dict_eps[key] = find_optimal_eps(dict_data_test[key])
        for key in dict_data_test.keys():
            print(f"\nAnalizzando {key}...")
            dbscan = DBSCAN(eps=best_dict_eps[key], min_samples=2)
            centroids_dict_dbscan[key] = dbscan.fit_predict(dict_data_test[key])
            n_clusters_ = len(set(centroids_dict_dbscan[key])) - (1 if -1 in centroids_dict_dbscan[key] else 0)
            n_noise_ = list(centroids_dict_dbscan[key]).count(-1)
            print("Estimated number of clusters: %d" % n_clusters_)
            print("Estimated number of noise points: %d" % n_noise_)
        dict_sorted_cluster = {}
        cluster_num = 2
        for key in centroids_dict_dbscan.keys():
            print(f"\nCompute the respective best cluster for {key} in feature space {feature}...")
            dict_sorted_cluster[key] = sorted_cluster(dict_data_test[key], centroids_dict_dbscan[key])
            plot_clusters(dict_data_test[key], centroids_dict_dbscan[key], title=f"Feature Space:{feature} for {key} - Top {cluster_num} Clusters", c=cluster_num, sorted_keys=dict_sorted_cluster[key])
            plot_clusters_with_images(dict_data_test[key],centroids_dict_dbscan[key],dict_image[key], f"Feature Space:{feature} for {key} - Top {cluster_num} Clusters with images", c=cluster_num, sorted_keys=dict_sorted_cluster[key])
        plot_all_datasets_together(dict_data_test, centroids_dict_dbscan, dict_sorted_cluster, c=cluster_num, title=f"Feature Space:{feature} for all labels - Top {cluster_num} Clusters")
        plot_all_datasets_with_images(dict_data_test, centroids_dict_dbscan, dict_image, dict_sorted_cluster, c=cluster_num, thumbnail_size=0.15, max_images_per_cluster=8, title=f"Feature Space:{feature} for all labels - Top {cluster_num} Clusters")










