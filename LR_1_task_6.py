import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("data_multivar_nb.txt", delimiter=",")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)
nb_accuracy = accuracy_score(y_test, nb_predictions)

print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))
print("Naive Bayes Classification Report:\n", classification_report(y_test, nb_predictions))
print(f"Accuracy SVM: {svm_accuracy:.4f}")
print(f"Accuracy Naive Bayes: {nb_accuracy:.4f}")

if svm_accuracy > nb_accuracy:
    print("SVM працює краще для цього набору даних.")
elif svm_accuracy < nb_accuracy:
    print("Наївний Байєс працює краще для цього набору даних.")
else:
    print("Обидві моделі мають однакову точність.")
