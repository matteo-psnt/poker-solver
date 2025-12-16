"""
Clustering utilities for card abstraction.

Implements k-means clustering for grouping poker hands with similar
equity distributions.
"""

import random
from typing import Tuple

import numpy as np


class KMeansClustering:
    """
    K-means clustering for card abstraction.

    Used to cluster hands with similar equity distributions into buckets.
    """

    def __init__(self, n_clusters: int, max_iters: int = 100, tolerance: float = 1e-4):
        """
        Initialize k-means clustering.

        Args:
            n_clusters: Number of clusters (buckets)
            max_iters: Maximum iterations for convergence
            tolerance: Convergence threshold (stop if centers move < tolerance)
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.cluster_centers = None
        self.labels = None

    def fit(self, data: np.ndarray, seed: int = 42) -> "KMeansClustering":
        """
        Fit k-means to data.

        Args:
            data: Data points (shape: [n_samples, n_features])
            seed: Random seed for initialization

        Returns:
            Self (for chaining)
        """
        random.seed(seed)
        np.random.seed(seed)

        n_samples = data.shape[0]

        if n_samples < self.n_clusters:
            raise ValueError(f"Cannot cluster {n_samples} samples into {self.n_clusters} clusters")

        # Initialize centers randomly (k-means++)
        self.cluster_centers = self._kmeans_plusplus_init(data)

        # Iterative clustering
        for iteration in range(self.max_iters):
            # Assign points to nearest cluster
            labels = self._assign_clusters(data)

            # Update cluster centers
            new_centers = self._update_centers(data, labels)

            # Check convergence
            center_shift = np.linalg.norm(new_centers - self.cluster_centers)
            self.cluster_centers = new_centers

            if center_shift < self.tolerance:
                break

        # Final assignment
        self.labels = self._assign_clusters(data)

        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Args:
            data: Data points (shape: [n_samples, n_features])

        Returns:
            Cluster labels (shape: [n_samples])
        """
        if self.cluster_centers is None:
            raise ValueError("Model not fitted yet")

        return self._assign_clusters(data)

    def predict_single(self, point: np.ndarray) -> int:
        """
        Predict cluster for a single point.

        Args:
            point: Single data point (shape: [n_features])

        Returns:
            Cluster label (int)
        """
        if self.cluster_centers is None:
            raise ValueError("Model not fitted yet")

        # Compute distances to all centers
        distances = np.linalg.norm(self.cluster_centers - point, axis=1)
        return int(np.argmin(distances))

    def _kmeans_plusplus_init(self, data: np.ndarray) -> np.ndarray:
        """
        K-means++ initialization for better starting centers.

        Args:
            data: Data points

        Returns:
            Initial cluster centers
        """
        n_samples = data.shape[0]
        centers = []

        # Choose first center randomly
        first_idx = random.randint(0, n_samples - 1)
        centers.append(data[first_idx])

        # Choose remaining centers
        for _ in range(1, self.n_clusters):
            # Compute distances to nearest center
            distances = np.array(
                [min(np.linalg.norm(point - center) for center in centers) for point in data]
            )

            # Square distances for k-means++ weighting
            squared_distances = distances**2
            probabilities = squared_distances / squared_distances.sum()

            # Sample next center with probability proportional to squared distance
            next_idx = np.random.choice(n_samples, p=probabilities)
            centers.append(data[next_idx])

        return np.array(centers)

    def _assign_clusters(self, data: np.ndarray) -> np.ndarray:
        """
        Assign each point to nearest cluster center.

        Args:
            data: Data points

        Returns:
            Cluster labels
        """
        # Compute distances to all centers
        distances = np.linalg.norm(
            data[:, np.newaxis, :] - self.cluster_centers[np.newaxis, :, :], axis=2
        )

        # Assign to nearest center
        return np.argmin(distances, axis=1)

    def _update_centers(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update cluster centers as mean of assigned points.

        Args:
            data: Data points
            labels: Current cluster assignments

        Returns:
            New cluster centers
        """
        new_centers = np.zeros_like(self.cluster_centers)

        for k in range(self.n_clusters):
            # Find points in this cluster
            cluster_points = data[labels == k]

            if len(cluster_points) > 0:
                # Update center as mean
                new_centers[k] = cluster_points.mean(axis=0)
            else:
                # Empty cluster - keep old center or reinitialize
                new_centers[k] = self.cluster_centers[k]

        return new_centers

    def get_inertia(self, data: np.ndarray) -> float:
        """
        Compute inertia (sum of squared distances to centers).

        Lower inertia = better clustering.

        Args:
            data: Data points

        Returns:
            Inertia value
        """
        if self.cluster_centers is None or self.labels is None:
            raise ValueError("Model not fitted yet")

        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = data[self.labels == k]
            if len(cluster_points) > 0:
                distances = np.linalg.norm(cluster_points - self.cluster_centers[k], axis=1)
                inertia += np.sum(distances**2)

        return inertia

    def __str__(self) -> str:
        """String representation."""
        return f"KMeansClustering(n_clusters={self.n_clusters})"


def find_optimal_k(
    data: np.ndarray, k_range: Tuple[int, int] = (5, 50), method: str = "elbow"
) -> int:
    """
    Find optimal number of clusters using elbow method.

    Args:
        data: Data to cluster
        k_range: Range of k values to try (min, max)
        method: Method to use ("elbow" or "silhouette")

    Returns:
        Optimal k value
    """
    if method != "elbow":
        raise NotImplementedError("Only elbow method implemented")

    k_min, k_max = k_range
    inertias = []
    k_values = list(range(k_min, k_max + 1))

    for k in k_values:
        kmeans = KMeansClustering(n_clusters=k)
        kmeans.fit(data)
        inertias.append(kmeans.get_inertia(data))

    # Find elbow point (maximum curvature)
    # Simple heuristic: find point with maximum distance to line
    # connecting first and last points

    # Normalize
    inertias = np.array(inertias)
    k_norm = np.linspace(0, 1, len(k_values))
    inertia_norm = (inertias - inertias.min()) / (inertias.max() - inertias.min())

    # Distance to line from first to last point
    line_points = np.column_stack([k_norm, inertia_norm])
    first_point = line_points[0]
    last_point = line_points[-1]

    distances = []
    for i, point in enumerate(line_points):
        # Distance from point to line
        distance = np.abs(np.cross(last_point - first_point, first_point - point))
        distance /= np.linalg.norm(last_point - first_point)
        distances.append(distance)

    # Find point with maximum distance (elbow)
    elbow_idx = np.argmax(distances)
    optimal_k = k_values[elbow_idx]

    return optimal_k
