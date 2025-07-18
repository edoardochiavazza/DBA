"""
– 10b: Implement a similar image search algorithm using this index structure storing the part
1 images and a visual model of your choice (the combined visual model must have at least
256 dimensions): for a given query image and integer t,
∗ visualizes the t most similar images,
∗ outputs the numbers of unique and overall number of images considered during the process
"""
import random

import numpy as np
from matplotlib import image as mpimg, pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler

from Task10.task10a import EuclideanLSH
from Task5 import task5
from Task6 import task6
from Task7.task7 import compute_pca_latent

if __name__ == '__main__':

    features_vect = ["hog_features", "cm10x10_features", "resnet_avgpool_1024_features", "resnet_fc_1000_features",
                     "resnet_layer3_1024_features"]
    n = 2
    random_n_feature_vect = random.sample(features_vect, n)
    data_train, labels_train, images_names_train = None, None, None
    print("")
    for i, feature in enumerate(random_n_feature_vect):
        # Carica i dati per questa feature
        data_feat, labels_feat, images_names_feat = task5.load_data_and_label_feature(
            feature, root_dir="../Task2/new_results"
        )

        if i == 0:
            # Prima iterazione: inizializza
            data_train = np.array(data_feat)
            labels_train = np.array(labels_feat)
            images_names_train = images_names_feat
        else:
            # Concatena orizzontalmente le features
            data_train = np.hstack((data_train, np.array(data_feat)))

    scaler = StandardScaler()
    data_train = scaler.fit_transform(data_train)
    k_95 = task6.pca_latent_semantics(data_train)
    data_reduced_pca, pca = compute_pca_latent(data_train, k_95)

    distances = pdist(data_reduced_pca, metric='euclidean') # calcolo tutte le distanze tratutti i puinti e faccio un vettore
    R = np.percentile(distances, 95) # calcolo la distanza che prende il 95 percento di queste distanze
    #R = 50
    print(R)
    L = 10  # numero di layer
    h = 5  # hash per layer
    d = data_reduced_pca.shape[1]  # dimensionalità

    # Crea indice LSH

    lsh = EuclideanLSH(L=L, h=h, d=d, r=R)

    # Aggiungi vettori
    lsh.add_vectors(data_reduced_pca)

    idx = np.random.randint(len(data_reduced_pca))
    query = data_reduced_pca[idx]
    print(f"Query primi 5 elementi: {query[:5]}")
    name_img = images_names_train[idx]
    label_img = labels_train[idx]
    path_img = "../Part1/" + label_img+"/"+name_img+".jpg"
    print("Nome immagine in input: " + str(name_img))
    img_input = mpimg.imread(path_img)
    plot = plt.imshow(img_input, cmap="grey")
    plt.show()
    print("Statistiche indice:")
    stats = lsh.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    print(f"Query primi 5 elementi: {query[:5]}")
    candidates, query_buckets, matches = lsh.search_with_info(query)

    print("BUCKET DELLA QUERY:")
    for item in query_buckets:
        print(f"Layer {item['layer']}: bucket {item['bucket']}")

    print(f"\nCANDIDATI TOTALI: {candidates}")

    print("\nMATCH TROVATI:")
    for item in matches:
        print(f"Layer {item['layer']}: bucket {item['bucket']}, candidati: {item['candidates']}")
    # Mostra primi 5 candidati ordinati per distanza
    candidates = lsh.query(query, return_distances=True)
    n = 10
    # Mostra primi n candidati ordinati per distanza
    if candidates:
        candidates.sort(key=lambda x: x[2])  # ordina per distanza
        print(f"\nTop {n} candidati più vicini:")
        for i, (vec_id, vec, dist) in enumerate(candidates[1:n+1]):
            name_img = images_names_train[vec_id]
            label_img = labels_train[vec_id]
            path_img = "../Part1/" + label_img+"/"+name_img+".jpg"
            print("Nome immagine in input: " + str(name_img))
            img_input = mpimg.imread(path_img)
            plot = plt.imshow(img_input, cmap="grey")
            plt.show()
            print(f"{i + 1}. Nome immagine {name_img}: distanza = {dist:.3f} prime 5 componenti = {vec[:5]}")
