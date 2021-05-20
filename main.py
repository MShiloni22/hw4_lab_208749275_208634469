from sys import argv
import os
from cross_validation import CrossValidation
from knn import KNN
from metrics import accuracy_score
from normalization import *


def load_data():
    """
    Loads data from path in first argument
    :return: returns data as list of Point
    """
    if len(argv) < 2:
        print('Not enough arguments provided. Please provide the path to the input file')
        exit(1)
    input_path = argv[1]

    if not os.path.exists(input_path):
        print('Input file does not exist')
        exit(1)

    points = []
    with open(input_path, 'r') as f:
        for index, row in enumerate(f.readlines()):
            row = row.strip()
            values = row.split(',')
            points.append(Point(str(index), values[:-1], values[-1]))
    return points


def run_knn(points):
    m = KNN(5)
    m.train(points)
    print(f'predicted class: {m.predict(points[0])}')
    print(f'true class: {points[0].label}')
    cv = CrossValidation()
    cv.run_cv(points, 10, m, accuracy_score, True, False)


"""
 def run_knn1(points):
    m = KNN(1)
    m.train(points)
    predicted = m.predict(points)
    true_labels = []
    for point in points:
        true_labels.append(point.label)
    print("accuracy_score for k=1:", accuracy_score(true_labels,predicted)) 
"""

def run_knn_k(points):
    best_classifier = 0
    best_accuracy_score = 0.0
    for k in range(1, 31):
        m = KNN(k)
        m.train(points)
        cv = CrossValidation()
        current_accuracy = cv.run_cv(points, len(points), m, accuracy_score, False, False)
        if current_accuracy > best_accuracy_score:
            best_accuracy_score = current_accuracy
            best_classifier = k
    return best_classifier


def question_3(points, k):
    m = KNN(k)
    m.train(points)
    n_folds_list = [2, 10, 20]
    print("Question 3:")
    print("K=", k, sep="")
    for i in n_folds_list:
        print(i, "-fold-cross-validation:", sep="")
        cv = CrossValidation()
        cv.run_cv(points, i, m, accuracy_score, False, True)


def question_4(points):
    k_list = [5, 7]
    normalization_list = [[DummyNormalizer, "DummyNormalizer"], [SumNormalizer, "SumNormalizer"],
                          [MinMaxNormalizer, "MinMaxNormalizer"], [ZNormalizer, "ZNormalizer"]]
    print("Question 4:")
    for k in k_list:
        print("K=", k, sep="")
        for i in normalization_list:
            normalize_object = i[0]()
            normalize_object.fit(points)
            new_points = normalize_object.transform(points)
            m = KNN(k)
            m.train(new_points)
            cv = CrossValidation()
            average_score = cv.run_cv(points, 2, m, accuracy_score, False, True)
            print("Accuracy of", i[1], "is", average_score, "\n")





if __name__ == '__main__':
    loaded_points = load_data()
    #  run_knn(loaded_points)
    #  run_knn1(loaded_points)
    best_k = run_knn_k(loaded_points)
    question_3(loaded_points, best_k)
    question_4(loaded_points)


