import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k, max_iterations=100, tolerance=1e-4):
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids = None

    def fit(self, data):
        # Inicializar los centroides seleccionando k puntos aleatorios del conjunto de datos
        self.centroids = data[np.random.choice(len(data), self.k, replace=False)]

        for _ in range(self.max_iterations):
            # Asignar cada punto al clúster cuyo centroide es el más cercano
            labels = self._assign_labels(data)

            # Calcular nuevos centroides basados en los puntos asignados a cada clúster
            new_centroids = self._compute_centroids(data, labels)

            # Verificar la convergencia
            if np.linalg.norm(new_centroids - self.centroids) < self.tolerance:
                break

            self.centroids = new_centroids

    def _assign_labels(self, data):
        # Calcular la distancia euclidiana de cada punto a cada centroide
        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)

        # Asignar cada punto al clúster con el centroide más cercano
        labels = np.argmin(distances, axis=1)

        return labels

    def _compute_centroids(self, data, labels):
        # Calcular nuevos centroides basados en los puntos asignados a cada clúster
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(self.k)])

        return new_centroids

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()
data = iris.data

# Normalizar los datos para facilitar la convergencia del algoritmo
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Crear una figura 
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Graficar datos antes de aplicar K-means
axes[0].scatter(data[:, 0], data[:, 1], c=iris.target, cmap='viridis', edgecolors='k')
axes[0].set_title('Datos originales')
axes[0].set_xlabel('Característica 1')
axes[0].set_ylabel('Característica 2')

# Crear y ajustar el modelo de K-means
kmeans = KMeans(k=3)
kmeans.fit(data)

# Obtener los centroides finales y las asignaciones de clúster
centroids = kmeans.centroids
labels = kmeans._assign_labels(data)

# Graficar datos después de aplicar K-means
axes[1].scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', edgecolors='k')
axes[1].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroides')
axes[1].set_title('Después de aplicar K-means')
axes[1].set_xlabel('Característica 1')
axes[1].set_ylabel('Característica 2')
axes[1].legend()

plt.tight_layout()
plt.show()
