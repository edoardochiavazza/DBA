"""
Task 10;
– 10a: Implement a Locality Sensitive Hashing (LSH) tool (for Euclidean distance) which
takes as input (a) the number of layers, L, (b) the number of hashes per layer, h, and (c) a set
of vectors as input and creates an in-memory index structure containing the given set of
vectors. See
”Near-Optimal Hashing Algorithms for Approximate Nearest Neighbor in High Dimensions”
(by Alexandr Andoni and Piotr Indyk). Communications of the ACM, vol. 51, no. 1, 2008,
pp. 117-122.
– 10b: Implement a similar image search algorithm using this index structure storing the part
1 images and a visual model of your choice (the combined visual model must have at least
256 dimensions): for a given query image and integer t,
∗ visualizes the t most similar images,
∗ outputs the numbers of unique and overall number of images considered during the process
"""
import pickle

import numpy as np
from collections import defaultdict
import random


class EuclideanLSH:
    def __init__(self, L, h, d, r=1.0):
        """
        LSH per distanza Euclidea
        Args:
            L: numero di layers (tabelle hash)
            h: numero di hash per layer
            d: dimensionalità dei vettori
            r: larghezza del bucket (influenza la sensibilità)
        """
        self.L = L
        self.h = h
        self.d = d
        self.r = r

        # Tabelle hash: L tabelle, ognuna con dizionario bucket -> lista vettori
        self.hash_tables = [defaultdict(list) for _ in range(L)]

        # Parametri random per ogni hash function
        # Ogni hash function ha vettore random a e offset b
        self.hash_params = []
        for layer in range(L):
            layer_params = []
            for hash_fn in range(h):
                # Vettore random da distribuzione normale
                a = np.random.normal(0, 1, d)
                # Offset random uniforme [0, r)
                b = random.uniform(0, r)
                layer_params.append((a, b))
            self.hash_params.append(layer_params)

        # Storage per i vettori originali
        self.vectors = []

    def _hash_vector(self, vector, layer_idx):
        """
        Calcola hash per un vettore in un layer specifico

        Hash function: h(v) = floor((a·v + b) / r)
        Combina h hash functions per layer
        """
        hashes = []
        for a, b in self.hash_params[layer_idx]:
            # Calcola hash: floor((a·v + b) / r)
            hash_val = int(np.floor((np.dot(a, vector) + b) / self.r))
            hashes.append(hash_val)

        # Combina gli h hash in una tupla (chiave del bucket)
        return tuple(hashes)

    def add_vectors(self, vectors):
        """
        Aggiunge vettori all'indice LSH

        Args:
            vectors: lista di vettori numpy
        """
        for vector in vectors:
            vector_id = len(self.vectors)
            self.vectors.append(vector)

            # Inserisci in ogni layer
            for layer_idx in range(self.L):
                bucket_key = self._hash_vector(vector, layer_idx)
                #print(f"Vector id: {vector_id} Layer id: {layer_idx} bucket key:", bucket_key)
                self.hash_tables[layer_idx][bucket_key].append(vector_id)

    def query(self, query_vector, return_distances=False):
        """
        Cerca vettori simili al query vector

        Args:
            query_vector: vettore di query
            return_distances: se True, restituisce anche le distanze

        Returns:
            lista di candidati (e distanze se richieste)
        """
        candidates = set()

        # Raccogli candidati da tutti i layer
        for layer_idx in range(self.L):
            bucket_key = self._hash_vector(query_vector, layer_idx)
            if bucket_key in self.hash_tables[layer_idx]:
                candidates.update(self.hash_tables[layer_idx][bucket_key])

        # Restituisci candidati
        candidate_vectors = [self.vectors[i] for i in candidates]

        if return_distances:
            distances = [np.linalg.norm(query_vector - vec) for vec in candidate_vectors]
            return list(zip(candidates, candidate_vectors, distances))
        else:
            return list(zip(candidates, candidate_vectors))

    def get_stats(self):
        """Restituisce statistiche sull'indice"""
        total_buckets = sum(len(table) for table in self.hash_tables)
        non_empty_buckets = sum(1 for table in self.hash_tables
                                for bucket in table.values() if bucket)

        return {
            'total_vectors': len(self.vectors),
            'layers': self.L,
            'hashes_per_layer': self.h,
            'total_buckets': total_buckets,
            'non_empty_buckets': non_empty_buckets,
            'avg_bucket_size': np.mean([len(bucket) for table in self.hash_tables
                                        for bucket in table.values() if bucket])
        }

    def print_bucket_contents(lsh_instance):
        """
        Stampa il contenuto di tutti i bucket in tutti i layer dell'istanza LSH

        Args:
            lsh_instance: un'istanza della classe EuclideanLSH
        """
        for layer_idx in range(lsh_instance.L):
            print(f"\nLayer {layer_idx + 1}/{lsh_instance.L}:")
            hash_table = lsh_instance.hash_tables[layer_idx]

            if not hash_table:
                print("  (Nessun bucket in questo layer)")
                continue

            for bucket_idx, (bucket_key, vector_ids) in enumerate(hash_table.items()):
                print(f"  Bucket {bucket_idx + 1} (key: {bucket_key}): {vector_ids}")

            print(f"  Totale bucket in questo layer: {len(hash_table)}")
            print(f"  Totale elementi in questo layer: {sum(len(v) for v in hash_table.values())}")

    def search_with_info(self, query_vector):
        candidates = set()
        query_buckets = []  # Tutti i bucket della query
        matches = []  # Solo i layer con match

        for layer_idx in range(self.L):
            bucket_key = self._hash_vector(query_vector, layer_idx)

            # Salva sempre il bucket della query per questo layer
            query_buckets.append({
                'layer': layer_idx,
                'bucket': bucket_key
            })

            # Se c'è un match, salva anche i candidati
            if bucket_key in self.hash_tables[layer_idx]:
                layer_candidates = self.hash_tables[layer_idx][bucket_key]
                candidates.update(layer_candidates)

                matches.append({
                    'layer': layer_idx,
                    'bucket': bucket_key,
                    'candidates': list(layer_candidates)
                })

        return candidates, query_buckets, matches

    def save_pickle(self, filepath):
        """
        Salva l'indice usando pickle

        Args:
            filepath: percorso del file (es. "my_lsh_index.pkl")
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Indice salvato in: {filepath}")

    @classmethod
    def load_pickle(cls, filepath):
        """
        Carica l'indice da file pickle

        Args:
            filepath: percorso del file

        Returns:
            istanza di EuclideanLSH caricata
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# Esempio d'uso
if __name__ == "__main__":
    # Parametri
    L = 5  # numero di layer
    h = 3  # hash per layer
    d = 10  # dimensionalità
    n = 1000  # numero di vettori

    # Genera vettori casuali
    vectors = [np.random.randn(d) for _ in range(n)]

    # Crea indice LSH
    lsh = EuclideanLSH(L=L, h=h, d=d, r=1.0)

    # Aggiungi vettori
    lsh.add_vectors(vectors)
    #lsh.print_bucket_contents()
    print("Statistiche indice:")
    stats = lsh.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    # Query
    query = np.random.randn(d)
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
    if candidates:
        candidates.sort(key=lambda x: x[2])  # ordina per distanza
        print("\nTop 5 candidati più vicini:")
        for i, (vec_id, vec, dist) in enumerate(candidates[:5]):
            print(f"  {i + 1}. Vettore {vec_id}: distanza = {dist:.3f} prime 5 componenti = {vec[:5]}")

    print("\nSalvataggio in pickle")
    lsh.save_pickle("my_lsh_index.pkl")


    print("\nCaricamento da pickle")
    lsh_loaded = EuclideanLSH.load_pickle("LSH_INDEX.pkl")

    print("Statistiche indice caricato:")
    stats_loaded = lsh_loaded.get_stats()
    for key, value in stats_loaded.items():
        print(f"  {key}: {value:.2f}")

    # Test query sull'indice caricato
    query = np.random.randn(d)
    candidates = lsh_loaded.query(query, return_distances=True)
    if candidates:
        candidates.sort(key=lambda x: x[2])
        print(f"\nQuery test: trovati {len(candidates)} candidati")
        print(f"Candidato più vicino: distanza = {candidates[0][2]:.3f}")

    # Verifica che gli indici siano identici
    print(f"\nVerifica identità: {np.array_equal(lsh.vectors[0], lsh_loaded.vectors[0])}")

    print("\nSalvataggio e caricamento pickle completati con successo!")


