from Task1 import task1 as t1
import os

if __name__ == "__main__":
    root_folder_image = "../Part1/brain_glioma"
    base_output_dir = "results"
    for dirpath, dirnames, filenames in os.walk(root_folder_image):
        for filename in filenames:
            # Percorso immagine completo
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
            all_features = t1.process_image_all_features(image_path, output_dir=output_dir)
