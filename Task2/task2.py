from Task1 import task1 as t1
import os
import numpy as np
from pathlib import Path

"""
Implement a program which extracts and stores feature descriptors for all images in
the data set.
"""

def check_npy_files(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith('.npy'):
                file_path = os.path.join(dirpath, file)
                try:
                    data = np.load(file_path)

                    if np.isnan(data).any():
                        print(f"Contiene NaN: {file_path}")
                    elif np.all(data == 0):
                        print(f"Tutti zeri: {file_path}")

                except Exception as e:
                    print(f"Errore nel file {file_path}: {e}")

if __name__ == "__main__":

    root_folder_image = "../Part2/"
    base_output_dir = "part2_results"
    for dirpath, dirnames, filenames in os.walk(root_folder_image):
        for filename in filenames:
            # Percorso immagine completo
            if filename == ".DS_Store" and os.path.abspath(dirpath) == os.path.abspath(root_folder_image):
                continue
            image_path = os.path.join(dirpath, filename)

            # Ricava il nome della sottocartella relativa a root_folder_image
            # es: se dirpath = Part1/sottocartella1
            # allora voglio "sottocartella1"
            relative_dir = os.path.relpath(dirpath, root_folder_image)
            direct = os.path.split(root_folder_image)[1]
            base = os.path.splitext(filename)[0]
            # Nome file senza estensione

            # Costruisci la cartella output combinando
            # results/relative_dir/nome_file_senza_estensione
            #output_dir = os.path.join(base_output_dir, relative_dir, name_wo_ext)
            output_dir = os.path.join(base_output_dir, direct)
            # Crea la cartella output se non esiste
            os.makedirs(output_dir, exist_ok=True)

            # Chiama la tua funzione con i percorsi corretti
            all_features = t1.process_image_all_features(image_path, output_dir=output_dir, visualize=False)

    check_npy_files("part2_results")

