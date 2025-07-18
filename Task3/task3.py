import os
import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from Task1 import task1 as t1

"""
: Implement a program which, given an image filename and a value “k”, returns and
visualizes the most similar k images based on each of the visual model -you will select the
appropriate distance/similarity measure for each feature model. For each match, also list the
corresponding distance/similarity score.
"""

"""
Distanza utilizzate: 
distanza euclidea per i cm: I Color Moments (tipicamente: media, deviazione, skewness per ciascun canale) sono features continue, dense e basse dimensionalità.
Usa distanza euclidea (L2), perché: le features hanno significato metrico; sono comparabili direttamente nel dominio numerico.

Gli HOG sono istogrammi (sparse, non negative, direzionali).
Due metriche adatte: Cosine Similarity :Ottima per confrontare la forma dell’istogramma, ignorando la scala.
Chi-quadro distance: Molto usata per istogrammi non normalizzati. Sensibile alle differenze relative tra i bin
cosine per resnet 

Resnet 50:
I vettori di resnet sono embedding semantici. L’informazione è nella direzione del vettore più che nella sua norma.
Cosine similarity è lo standard in image retrieval e visual search
"""

def compute_k_search(image_path, k, distance_metrics):

    similarity_dict_cm ={}
    similarity_dict_hog ={}
    similarity_dict_resnet_avg ={}
    similarity_dict_resnet_fc = {}
    similarity_dict_resnet_layer3 = {}
    root_folder_image = "../Task2/new_results"

    dict_input_image = t1.process_image_all_features(image_path, output_dir=None, visualize=False)
    for dirpath, dirnames, filenames in os.walk(root_folder_image):
        image_name = os.path.basename(dirpath)
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            if filename == "cm10x10_features.npy":
                feature_vect_npy=np.load(full_path)
                vect_input = dict_input_image["color_moments"]
                if distance_metrics == 'euclidean':
                    distance = np.linalg.norm(vect_input - feature_vect_npy)
                    similarity_dict_cm[image_name] = distance
                elif distance_metrics == 'cosine':
                    pass
            elif filename == "hog_features.npy":
                hog_npy=np.load(full_path)
                vect_input = dict_input_image["hog_features"]
                # Chi quadro con epsilon per stabilità numerica
                if distance_metrics == 'euclidean':
                    epsilon = 1e-10
                    similarity_dict_hog[image_name] = 0.5 * np.sum((vect_input - hog_npy) ** 2 / (vect_input + hog_npy + epsilon)) # chi quadro
                elif distance_metrics == 'cosine':
                    similarity_dict_hog[image_name] = np.dot(vect_input, hog_npy) / (np.linalg.norm(vect_input) * np.linalg.norm(hog_npy))   #cosine similarity
            elif filename =="resnet_layer3_1024_features.npy":
                resnet_layer3_npy = np.load(full_path)
                vect_input = dict_input_image["resnet_layer3_1024"]
                if distance_metrics == 'cosine':
                    similarity_dict_resnet_layer3[image_name] = np.dot(vect_input, resnet_layer3_npy) / (np.linalg.norm(vect_input) * np.linalg.norm(resnet_layer3_npy))
            elif filename == "resnet_avgpool_1024_features.npy":
                resnet_avg_npy=np.load(full_path)
                vect_input = dict_input_image["resnet_avgpool_1024"]
                if distance_metrics == 'cosine':
                    similarity_dict_resnet_avg[image_name] = np.dot(vect_input, resnet_avg_npy) / (np.linalg.norm(vect_input) * np.linalg.norm(resnet_avg_npy))
                elif distance_metrics == 'euclidean':
                    pass
            elif filename =="resnet_fc_1000_features.npy":
                resnet_fc_npy=np.load(full_path)
                vect_input = dict_input_image["resnet_fc_1000"]
                if distance_metrics == 'cosine':
                    similarity_dict_resnet_fc[image_name] = np.dot(vect_input, resnet_fc_npy) / (np.linalg.norm(vect_input) * np.linalg.norm(resnet_fc_npy))
                elif distance_metrics == 'euclidean':
                    pass
            else:
               print("unknown file or feature vector")


    similarity_dict_resnet_fc = sorted(similarity_dict_resnet_fc.items(), key=lambda item: item[1], reverse=True)[1:k+1]
    similarity_dict_resnet_avg = sorted(similarity_dict_resnet_avg.items(), key=lambda item: item[1], reverse=True)[1:k+1]
    similarity_dict_resnet_layer3 = sorted(similarity_dict_resnet_layer3.items(), key=lambda item: item[1], reverse=True)[1:k+1]
    similarity_dict_hog = sorted(similarity_dict_hog.items(), key=lambda item: item[1])[1:k+1]
    similarity_dict_cm = sorted(similarity_dict_cm.items(), key=lambda item: item[1], reverse=True)[1:k + 1]

    return similarity_dict_cm, similarity_dict_hog, similarity_dict_resnet_layer3, similarity_dict_resnet_fc, similarity_dict_resnet_avg

def show_k_images(sim_dict, feature, metrica, image_input):
    print("Nome immagine in input: " + str(os.path.basename(image_input)))
    img_input = mpimg.imread(image_input)
    plt.imshow(img_input, cmap="grey")
    plt.show()
    for image_name, sim_value in sim_dict:
        print("Nome immagine: " +  str(image_name))
        print("Feature: " + feature)
        print("Similarity valore: " + str(sim_value) + " calcolato con metrica " + metrica)
        base_name = image_name.rsplit("_", 1)[0]
        full_path = os.path.join("../Part1/",base_name + "/" + image_name + ".jpg")
        img = mpimg.imread(full_path)
        plt.imshow(img, cmap="grey")
        plt.show()
        print()


if __name__ == '__main__':

    print("write a integer k value for visualize the k similiar images of the input image")
    #input_k = input()
    input_k = 4
    #print("write a name for the input image")
    #input_name = input()
    image_p = "../Part1/brain_glioma/brain_glioma_0001.jpg"
    distance_metrics = ['cosine', 'euclidean']
    random_distance_metrics = random.sample(distance_metrics, k=1)
    sm_dict_cm, sm_dict_hog, sm_dict_resnet_layer3, sm_dict_resnet_fc, sm_dict_resnet_avg = compute_k_search(image_p, input_k,random_distance_metrics)
    print("Color moments results:")
    show_k_images(sm_dict_cm, "Color_moments",f"{random_distance_metrics}", image_p)
    print("HOG results:")
    show_k_images(sm_dict_hog, "Histogram of gradient", f"{random_distance_metrics}",image_p)
    print("Resnet layer3 results:")
    show_k_images(sm_dict_resnet_layer3, "ResNet layer3", f"{random_distance_metrics}",image_p)
    print("Resnet FC results:")
    show_k_images(sm_dict_resnet_fc, "ResNet fc", f"{random_distance_metrics}", image_p)
    print("Resnet avgpool results:")
    show_k_images(sm_dict_resnet_avg, "ResNet AvgPool", f"{random_distance_metrics}", image_p)

