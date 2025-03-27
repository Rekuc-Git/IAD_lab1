import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


def smirniv_confusion_matrix(y_true, y_pred):
    def find_TP(y_true, y_pred):
        return sum((y_true == 1) & (1 == y_pred))

    def find_FN(y_true, y_pred):
        return sum((y_true == 1) & (0 == y_pred))

    def find_FP(y_true, y_pred):
        return sum((y_true == 0) & (1 == y_pred))

    def find_TN(y_true, y_pred):
        return sum((y_true == 0) & (0 == y_pred))

    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)

    return np.array([[TN, FP], [FN, TP]])


def smirniv_accuracy_score(y_true, y_pred):
    TN, FP, FN, TP = smirniv_confusion_matrix(y_true, y_pred).ravel()
    return (TP + TN) / (TP + TN + FP + FN)


def smirniv_recall_score(y_true, y_pred):
    TN, FP, FN, TP = smirniv_confusion_matrix(y_true, y_pred).ravel()
    return TP / (TP + FN) if (TP + FN) != 0 else 0.0


def smirniv_precision_score(y_true, y_pred):
    TN, FP, FN, TP = smirniv_confusion_matrix(y_true, y_pred).ravel()
    return TP / (TP + FP) if (TP + FP) != 0 else 0.0


def smirniv_f1_score(y_true, y_pred):
    recall = smirniv_recall_score(y_true, y_pred)
    precision = smirniv_precision_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0


# Завантаження даних
df = pd.read_csv('data.csv')
thresh = 0.5
df['predicted_RF'] = (df.model_RF >= thresh).astype(int)
df['predicted_LR'] = (df.model_LR >= thresh).astype(int)

# Перевірка реалізованих функцій
assert np.array_equal(smirniv_confusion_matrix(df.actual_label.values, df.predicted_RF.values),
                      confusion_matrix(df.actual_label.values, df.predicted_RF.values)), 'Error in confusion_matrix()'
assert np.isclose(smirniv_accuracy_score(df.actual_label.values, df.predicted_RF.values),
                  accuracy_score(df.actual_label.values, df.predicted_RF.values)), 'Error in accuracy_score()'
assert np.isclose(smirniv_recall_score(df.actual_label.values, df.predicted_RF.values),
                  recall_score(df.actual_label.values, df.predicted_RF.values)), 'Error in recall_score()'
assert np.isclose(smirniv_precision_score(df.actual_label.values, df.predicted_RF.values),
                  precision_score(df.actual_label.values, df.predicted_RF.values)), 'Error in precision_score()'
assert np.isclose(smirniv_f1_score(df.actual_label.values, df.predicted_RF.values),
                  f1_score(df.actual_label.values, df.predicted_RF.values)), 'Error in f1_score()'

# Вивід результатів
print(f'Accuracy RF: {smirniv_accuracy_score(df.actual_label.values, df.predicted_RF.values):.3f}')
print(f'Recall RF: {smirniv_recall_score(df.actual_label.values, df.predicted_RF.values):.3f}')
print(f'Precision RF: {smirniv_precision_score(df.actual_label.values, df.predicted_RF.values):.3f}')
print(f'F1 Score RF: {smirniv_f1_score(df.actual_label.values, df.predicted_RF.values):.3f}')
