import numpy as np
from sklearn.preprocessing import Binarizer, StandardScaler, MinMaxScaler, Normalizer

input_data = np.array([2.5, -1.6, -6.1, -2.4, -1.2, 4.3, 3.2, 3.1, 6.1, -4.4, 1.4, -1.2, 2.5]).reshape(-1, 1)

binarizer = Binarizer(threshold=0)
binarized_data = binarizer.fit_transform(input_data)
print("\nВхідні данні:")
print(input_data.flatten())
print("\nБінаризовані дані:")
print(binarized_data.T)

scaler = StandardScaler()
mean_excluded_data = scaler.fit_transform(input_data)
print("\nДані після виключення середнього:")
print(mean_excluded_data.T)

minmax_scaler = MinMaxScaler()
scaled_data = minmax_scaler.fit_transform(input_data)
print("\nМасштабовані дані:")
print(scaled_data.T)

normalizer = Normalizer()
normalized_data = normalizer.fit_transform(input_data.T)
print("\nНормалізовані дані:")
print(normalized_data)
