from matplotlib import image as mpimg, pyplot as plt

from Task1 import task1 as t1
import os
import numpy as np
from sklearn.metrics.pairwise import nan_euclidean_distances

"""
Implement a program which, given (a) a part 2 query image file, (b) a user selected feature
space, and (c) positive integer k (k<=2), identifies and lists k most likely matching labels,
along with their scores, under the selected feature space
"""

def compute_average_all_features_space_for_all_labels():
    root_folder_image = "../Task2/new_results"
    similarity_dict_cm = {}
    similarity_dict_hog = {}
    similarity_dict_resnet_avg = {}
    similarity_dict_resnet_fc = {}
    similarity_dict_resnet_layer3 = {}
    dict_list = [(similarity_dict_cm,"cm"), (similarity_dict_hog,"hog"), (similarity_dict_resnet_avg,"rn_avg"), (similarity_dict_resnet_fc,"rn_fc"), (similarity_dict_resnet_layer3,"rn_l3")]
    num_labels = 1002
    for dirpath, dirnames, filenames in os.walk(root_folder_image):
        dir_image_name = os.path.basename(dirpath)
        dir_label_name = dir_image_name.rsplit("_", 1)[0]
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            if filename == "cm10x10_features.npy":
                feature_vect_npy=np.load(full_path)
                if dir_label_name not in similarity_dict_cm:
                    similarity_dict_cm[dir_label_name] = np.zeros_like(feature_vect_npy)
                similarity_dict_cm[dir_label_name] += feature_vect_npy  #cosine similarity
            elif filename == "hog_features.npy":
                hog_npy=np.load(full_path)
                # Chi quadro con epsilon per stabilità numerica
                epsilon = 1e-10
                if dir_label_name not in similarity_dict_hog:
                    similarity_dict_hog[dir_label_name] = np.zeros_like(hog_npy)
                similarity_dict_hog[dir_label_name] += hog_npy# chi quadro
            elif filename =="resnet_layer3_1024_features.npy":
                resnet_layer3_npy = np.load(full_path)
                if dir_label_name not in similarity_dict_resnet_layer3:
                    similarity_dict_resnet_layer3[dir_label_name] = np.zeros_like(resnet_layer3_npy)
                similarity_dict_resnet_layer3[dir_label_name] += resnet_layer3_npy
            elif filename == "resnet_avgpool_1024_features.npy":
                resnet_avg_npy = np.load(full_path)
                if dir_label_name not in similarity_dict_resnet_avg:
                    similarity_dict_resnet_avg[dir_label_name] = np.zeros_like(resnet_avg_npy)
                similarity_dict_resnet_avg[dir_label_name] += resnet_avg_npy
            elif filename =="resnet_fc_1000_features.npy":
                resnet_fc_npy=np.load(full_path)
                if dir_label_name not in similarity_dict_resnet_fc:
                    similarity_dict_resnet_fc[dir_label_name] = np.zeros_like(resnet_fc_npy)
                similarity_dict_resnet_fc[dir_label_name] += resnet_fc_npy
            else:
               print("unknown file or feature vector")
    for d in dict_list:
        for key in d[0]:
            d[0][key] = d[0][key] / num_labels
            full_path = os.path.join("new_avg_labels/", key)
            full_path = full_path +"_"+ d[1]
            np.save(full_path, d[0][key])


def load_avg_feature_vector(feature_type):
    avg_vect_glioma, avg_vect_menin, avg_vect_tumor = [[] for _ in range(3)]
    if feature_type == "cm":
        avg_vect_glioma =  np.load("new_avg_labels/brain_glioma/brain_glioma_cm.npy")
        avg_vect_menin = np.load("new_avg_labels/brain_menin/brain_menin_cm.npy")
        avg_vect_tumor = np.load("new_avg_labels/brain_tumor/brain_tumor_cm.npy")
    elif feature_type == "hog":
        avg_vect_glioma = np.load("new_avg_labels/brain_glioma/brain_glioma_hog.npy")
        avg_vect_menin = np.load("new_avg_labels/brain_menin/brain_menin_hog.npy")
        avg_vect_tumor = np.load("new_avg_labels/brain_tumor/brain_tumor_hog.npy")
    elif feature_type == "rn_avg":
        avg_vect_glioma = np.load("new_avg_labels/brain_glioma/brain_glioma_rn_avg.npy")
        avg_vect_menin = np.load("new_avg_labels/brain_menin/brain_menin_rn_avg.npy")
        avg_vect_tumor = np.load("new_avg_labels/brain_tumor/brain_tumor_rn_avg.npy")
    elif feature_type == "rn_l3":
        avg_vect_glioma = np.load("new_avg_labels/brain_glioma/brain_glioma_rn_l3.npy")
        avg_vect_menin = np.load("new_avg_labels/brain_menin/brain_menin_rn_l3.npy")
        avg_vect_tumor = np.load("new_avg_labels/brain_tumor/brain_tumor_rn_l3.npy")
    elif feature_type == "rn_fc":
        avg_vect_glioma = np.load("new_avg_labels/brain_glioma/brain_glioma_rn_fc.npy")
        avg_vect_menin = np.load("new_avg_labels/brain_menin/brain_menin_rn_fc.npy")
        avg_vect_tumor = np.load("new_avg_labels/brain_tumor/brain_tumor_rn_fc.npy")
    return  avg_vect_glioma, avg_vect_menin, avg_vect_tumor

def compute_top_k_score(k, dict_input_image, avg_vect_glioma, avg_vect_menin, avg_vect_tumor, feature_type):

    ranking_euclidean_distances = {}
    ranking_cosine_similarity = {}
    input_image_ft_vect = dict_input_image[feature_type]
    glioma_sim = np.dot(input_image_ft_vect, avg_vect_glioma) / (np.linalg.norm(input_image_ft_vect) * np.linalg.norm(avg_vect_glioma))
    tumor_sim = np.dot(input_image_ft_vect, avg_vect_tumor) / (np.linalg.norm(input_image_ft_vect) * np.linalg.norm(avg_vect_tumor))
    menin_sim = np.dot(input_image_ft_vect, avg_vect_menin) / (np.linalg.norm(input_image_ft_vect) * np.linalg.norm(avg_vect_menin))
    ranking_cosine_similarity["glioma"] = glioma_sim
    ranking_cosine_similarity["tumor"] = tumor_sim
    ranking_cosine_similarity["menin"] = menin_sim
    glioma_sim= np.linalg.norm(input_image_ft_vect - avg_vect_glioma)
    tumor_sim = np.linalg.norm(input_image_ft_vect - avg_vect_tumor)
    menin_sim = np.linalg.norm(input_image_ft_vect - avg_vect_menin)
    ranking_euclidean_distances["glioma"] = glioma_sim
    ranking_euclidean_distances["tumor"] = tumor_sim
    ranking_euclidean_distances["menin"] = menin_sim

    ranking_euclidean_distances = sorted(ranking_euclidean_distances.items(), key=lambda item: item[1], reverse=False)[0:k]
    ranking_cosine_similarity = sorted(ranking_cosine_similarity.items(), key=lambda item: item[1], reverse=True)[0:k]

    return ranking_euclidean_distances, ranking_cosine_similarity

def print_ranking_info(image_in ,ranking_euclidean_distances, ranking_cosine_similarity, feature_type):

    print("Nome immagine in input: " + str(os.path.basename(image_in)))
    img_input = mpimg.imread(image_in)
    plot = plt.imshow(img_input, cmap="grey")
    plt.show()

    for label, sim_value in ranking_euclidean_distances:
        print("Label: " + label)
        print("Feature: " + feature_type)
        print("Similarity valore: " + str(sim_value) + " calcolato con metrica distanza euclidea")
        print()
    for label, sim_value in ranking_cosine_similarity:
        print("Label: " + label)
        print("Feature: " + feature_type)
        print("Similarity valore: " + str(sim_value) + " calcolato con metrica similarità del coseno")
        print()



if __name__ == '__main__':

    #compute_average_all_features_space_for_all_labels()
    avg_vect_glioma_i, avg_vect_menin_i, avg_vect_tumor_i = load_avg_feature_vector("cm")
    image_input = "../Part2/brain_menin/brain_menin_1617.jpg"
    dict_input_image_i = t1.process_image_all_features(image_input, output_dir=None)
    #in_k = input("seleziona un k (k<=2)")
    in_k = 2
    ft = "color_moments"
    ranking_euclidean_distances_i, ranking_cosine_similarity_i = compute_top_k_score(in_k, dict_input_image_i, avg_vect_glioma_i, avg_vect_menin_i, avg_vect_tumor_i, ft)
    print_ranking_info(image_input, ranking_euclidean_distances_i,ranking_cosine_similarity_i, ft)
