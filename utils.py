import numpy as np
from typing import Callable

from sklearn.cluster import AgglomerativeClustering


def condense_labels(labels: np.ndarray, embedding_func: Callable, threshold: float=0.5):

    embeddings = np.array(embedding_func(labels))
    
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=threshold
    ).fit(embeddings)

    clusters = [np.where(clustering.labels_ == l)[0] 
                for l in range(clustering.n_clusters_)]

    clusters_reduced = []
    
    for c in clusters:
        embs = embeddings[c]
        centroid = np.mean(embs)

        idx = c[np.argmin(np.linalg.norm(embs - centroid, axis=1))]
        clusters_reduced.append(idx)

    old2new = {old_id: new_id for old_ids, new_id in zip(clusters, clusters_reduced) for old_id in old_ids}
    
    return {labels[i]: labels[j] for i, j in old2new.items()}