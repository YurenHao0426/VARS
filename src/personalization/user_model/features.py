import numpy as np
from dataclasses import dataclass
from sklearn.decomposition import PCA

@dataclass
class ItemProjection:
    P: np.ndarray     # [k, d]
    mean: np.ndarray  # [d]

    @classmethod
    def from_pca(cls, embeddings: np.ndarray, k: int) -> "ItemProjection":
        """
        embeddings: [M, d]
        """
        mean = embeddings.mean(axis=0)
        centered = embeddings - mean
        
        # Ensure k is not larger than min(n_samples, n_features)
        n_samples, n_features = embeddings.shape
        actual_k = min(k, n_samples, n_features)
        
        pca = PCA(n_components=actual_k)
        pca.fit(centered)

        # pca.components_: [k, d]
        P = pca.components_  # Each row is a principal component vector
        
        # If we had to reduce k, we might want to pad P or handle it?
        # For now, let's assume we get what we asked for or less if data is small.
        # But for the system we want fixed k. 
        # If actual_k < k, we should pad with zeros to match expected dimension.
        if actual_k < k:
            padding = np.zeros((k - actual_k, n_features), dtype=P.dtype)
            P = np.vstack([P, padding])
            
        return cls(P=P, mean=mean)

    def transform_embeddings(self, E: np.ndarray) -> np.ndarray:
        """
        E: [N, d] -> [N, k]
        """
        return (E - self.mean) @ self.P.T

    def transform_vector(self, e: np.ndarray) -> np.ndarray:
        """
        e: [d] -> [k]
        """
        return self.P @ (e - self.mean)

