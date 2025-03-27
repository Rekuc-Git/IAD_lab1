import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def visualize_classifier(classifier, X, y):
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    mesh_step_size = 0.01

    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size),
                                 np.arange(min_y, max_y, mesh_step_size))

    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)

    cmap_background = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA', '#FFD700'])
    cmap_points = ListedColormap(['#FF0000', '#0000FF', '#00FF00', '#FFA500'])

    plt.contourf(x_vals, y_vals, output, cmap=cmap_background, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_points, edgecolors='black', linewidth=1, marker='o', s=80)

    plt.xlabel('Ознака 1')
    plt.ylabel('Ознака 2')
    plt.title('Результати класифікації')
    plt.show()
