from Task1 import task1 as t1
import os
import numpy as np
"""
Implement a program which, given (a) a part 2 query image file, (b) a user selected feature
space, and (c) positive integer k (k<=2), identifies and lists k most likely matching labels,
along with their scores, under the selected feature space
"""

def compute_average_all_features_space_for_all_labels():
    root_folder_image = "../Task2/results"
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
            full_path = os.path.join("avg_labels/", key)
            full_path = full_path +"_"+ d[1]
            np.save(full_path, d[0][key])

if __name__ == '__main__':
    compute_average_all_features_space_for_all_labels()

