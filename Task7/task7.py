from pyexpat import features

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from Task5 import task5
from Task6 import task6
import numpy as np

"""
Task 7: Implement a program which,
– for each unique label l, computes the corresponding k latent semantics (of your choice)
associated with the part 1 images, and
– for the part 2 images, predicts the most likely labels using distances/similarities computed
under the label-specific latent semantics.
The system should also output per-label precision, recall, and F1-score values as well as
output an overall accuracy value.
"""

"""
idea: spazio comune: prova rgb e hog con svd e clasfficatori(k-nn e svm), lda da solo. come k cercherò il num che mi faccia perdere al massimo 5% di informazione
      spazio per label: riduco dimensioni con (svd o pca) su rgb e hog con 100 come k di latente semantics e poi cerco le nuove immagini a quale spazio si avvicina di più. 
"""
def compute_pca_latent(data, k):
    print("Performing PCA...")
    pca = PCA(k)
    data_reduced = pca.fit_transform(data)
    return data_reduced,pca

def train_knn(data, labels, num_neighbors = 5):
    knn = KNeighborsClassifier(n_neighbors= num_neighbors)
    knn.fit(data, labels)
    return knn

def train_svm(data, labels):
    svm = SVC(kernel='rbf')  # o 'linear'
    svm.fit(data, labels)
    return svm

def classify_knn(knn, data):
    predictions = knn.predict(data)
    return predictions

def classify_svm(svm, data):
    predictions = svm.predict(data)
    return predictions

def classify_lda(lda, data):
    predictions = lda.predict(data)
    return predictions

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def multiclass_metrics_macro_verbose(y_true, y_pred, labels=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))

    precisions, recalls, f1s = [], [], []

    print("Per-class metrics:")
    print("-" * 30)

    for label in labels:
        TP = np.sum((y_pred == label) & (y_true == label))
        FP = np.sum((y_pred == label) & (y_true != label))
        FN = np.sum((y_pred != label) & (y_true == label))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        print(f"Class {label}: TP={TP}, FP={FP}, FN={FN}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print()

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    accuracy = np.sum(y_true == y_pred) / len(y_true)

    print("Overall metrics:")
    print("-" * 30)
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")
    print(f"Macro F1-score:  {macro_f1:.4f}")
    print(f"Accuracy:        {accuracy:.4f}")

    return macro_precision, macro_recall, macro_f1, accuracy

def compute_reduction_classifing_metrics(data_test, data_train ,labels_test, labels_train):
    k_svd = task6.print_inherith_dimension_svd(data_train)
    print(f"using best k value for svd:{k_svd}")
    data_reduced_train, _, Vt = task5.svd_latent_semantics(data_train, k_svd)
    #print(data_reduced.shape)
    knn = train_knn(data_reduced_train, labels_train)
    svm = train_svm(data_reduced_train, labels_train)
    print("Classification using knn :")
    data_test_reduced = data_test @ Vt[:k_svd].T
    predictions = classify_knn(knn, data_test_reduced)
    multiclass_metrics_macro_verbose(labels_test, predictions, np.unique(labels_test))
    print("-" * 60)
    print("Classification using svm :")
    predictions = classify_svm(svm, data_test_reduced)
    multiclass_metrics_macro_verbose(labels_test, predictions, np.unique(labels_test))
    print("-" * 60)
    print("Reduction and classification using LDA: ")
    data_reduced_train, lda = task5.lda_latent_semantics(data_train, labels_train, k=2)
    data_test_reduced = lda.transform(data_test)
    lda.fit(data_test_reduced, labels_test)
    predictions = classify_lda(lda, data_test_reduced)
    multiclass_metrics_macro_verbose(labels_test, predictions, np.unique(labels_test))
    print("-" * 60)

if __name__ == '__main__':

    # rgb
    # Spazio comune
    path_part1 = "../Part1"
    path_part2 = "../Part2"
    print("Reduce and classification using all the illness brain vectors together")
    features_vect = ["hog_features", "cm10x10_features", "resnet_avgpool_1024_features", "resnet_fc_1000_features", "resnet_layer3_1024_features", "rgb"]
    for feature in features_vect:
        print(f"Using feature space: {feature}")
        print(f"Loading training data for {feature}...")
        if feature == "rgb":
            data_train, labels_train = task6.load_images(path_part1, target_size=(224, 224))
            data_test, labels_test = task6.load_images(path_part2, target_size=(224, 224))
        else:
            data_train, labels_train, images_names_train = task5.load_data_and_label_feature(feature, root_dir="../Task2/new_results")
            data_test, labels_test, images_names_test = task5.load_data_and_label_feature(feature,root_dir="../Task2/part2_results")
        print(f"Loading test data for {feature}...")
        compute_reduction_classifing_metrics(data_test, data_train ,labels_test, labels_train)
    print("Reduce the illness brain vectors one for one and use cosine similarity for compute similarity (cs bc is in)")
    for feature in features_vect:
        print(f"Using feature space: {feature}")
        if feature == "rgb":
            data_train, labels_train = task6.load_images(path_part1, target_size=(224, 224))
            data_test, labels_test = task6.load_images(path_part2, target_size=(224, 224))
        else:
            data_train, labels_train, images_names_train = task5.load_data_and_label_feature(feature, root_dir="../Task2/new_results")
            data_test, labels_test, images_names_test = task5.load_data_and_label_feature(feature,root_dir="../Task2/part2_results")
        data_train = np.array(data_train)
        labels_train = np.array(labels_train)
        dict_data_test = {"tumor": data_train[labels_train == "brain_tumor"], "menin": data_train[labels_train == "brain_menin"], "glioma": data_train[labels_train == "brain_glioma"]}
        predictions_test_cosine = []
        predictions_test_euclidean = []
        centroids_dict_test_pca = {}
        for key in dict_data_test.keys():
            data_reduced_pca, pca = compute_pca_latent(dict_data_test[key], 100)
            centroid_pca = data_reduced_pca.mean(axis=0)
            centroids_dict_test_pca[key] = (centroid_pca, pca)
        for i in range(0, len(data_test)):
            img = data_test[i].reshape(1, -1)
            label = labels_test[i]
            dict_sim_pca_cosine = {}
            dict_sim_pca_euclidean = {}
            for key in centroids_dict_test_pca:
                pca = centroids_dict_test_pca[key][1]
                centroid_pca_label = centroids_dict_test_pca[key][0]
                img_test_pca = pca.transform(img)
                dict_sim_pca_cosine[key] = cosine_similarity(img_test_pca,  centroid_pca_label)
                dict_sim_pca_euclidean[key]   = np.sqrt(np.sum((img_test_pca - centroid_pca_label) ** 2))
            max_key_cosine = max(dict_sim_pca_cosine, key=dict_sim_pca_cosine.get)
            max_key_eucledian = max(dict_sim_pca_euclidean, key=dict_sim_pca_euclidean.get)
            print("Name image: ", images_names_test[i],"Prediction cosine similarity: ", max_key_cosine ," Prediction euclidean similarity for centroid pca: ", max_key_eucledian, " Truth label: ", label)
            predictions_test_euclidean.append("brain_"+max_key_eucledian)
            predictions_test_cosine.append("brain_"+max_key_cosine)
        print("Metrics compute using cosine similarity:")
        multiclass_metrics_macro_verbose(labels_test, predictions_test_cosine, np.unique(labels_test))
        print("-" * 60)
        print("Metrics compute using euclidean distance:")
        multiclass_metrics_macro_verbose(labels_test, predictions_test_euclidean, np.unique(labels_test))
        print("-" * 300)























