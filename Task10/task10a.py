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

    # Query
    query = np.random.randn(d)
    candidates = lsh.query(query, return_distances=True)

    print(f"Query trovato {len(candidates)} candidati")
    print("Statistiche indice:")
    stats = lsh.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Mostra primi 5 candidati ordinati per distanza
    if candidates:
        candidates.sort(key=lambda x: x[2])  # ordina per distanza
        print("\nTop 5 candidati più vicini:")
        for i, (vec_id, vec, dist) in enumerate(candidates[:5]):
            print(f"  {i + 1}. Vettore {vec_id}: distanza = {dist:.3f}")




