# -*- coding: utf-8 -*-
from itertools import product
import pandas as pd
import numpy as np
from scipy.io.arff import loadarff
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC

TEST_FRACTION = 0.2
MAX_CPU_TO_USE = 8
FOLDS_NUMBER = 4
METRICS_COUNT = 4
SEED = 42
LATEX = False

DATASETS = [
    ('breast_cancer', '../data/dataset_13_breast-cancer.arff'),
    ('nursery', '../data/dataset_26_nursery.arff'),
    ('diabetes', '../data/dataset_37_diabetes.arff'),
    ('iris', '../data/dataset_61_iris.arff'),
]


CLASSIFIERS = [
    (KNeighborsClassifier, [{'n_neighbors': k} for k in xrange(3, 7)]),
    (LogisticRegression, [{'C': c} for c in 10.0 ** np.arange(-1, 3)]),
    (SVC, [{'kernel': 'rbf', 'C': c, 'gamma': gamma, 'probability': True} for c, gamma in product(
        [10.0, 1000.0],  # C
        [0.1, 0.01],     # gamma
    )]),
]


if __name__ == '__main__':
    np.random.seed(SEED)
    vectorizer = DictVectorizer(sparse=False)
    label_encoder = LabelEncoder()
    normalizer = MinMaxScaler()
    scaler = StandardScaler()
    frames = []

    for name, path in DATASETS:
        print
        print '=' * 40
        print
        print 'DATASET:', name

        # loading data
        data, _ = loadarff(path)
        df = pd.DataFrame.from_records(data).rename(columns=str.lower)

        # build classes
        y = label_encoder.fit_transform(df['class'])

        # build features
        x = df.drop('class', axis=1).to_dict('records')

        # transform nominal features
        x = vectorizer.fit_transform(x)

        # scaling
        x = normalizer.fit_transform(x)

        # build folds indexes
        folds = list(StratifiedKFold(y, FOLDS_NUMBER, shuffle=True, random_state=SEED))

        context = np.zeros((y.shape[0], len(CLASSIFIERS)))
        best_parameters = []

        class_count = np.unique(y).shape[0]

        for classifier_index, classifier_and_params in enumerate(CLASSIFIERS):
            Classifier, parameters_configurations = classifier_and_params
            print Classifier.__name__

            max_f1 = 0.0
            max_i = -1

            context_column_variants = np.zeros((y.shape[0], len(parameters_configurations)), dtype=float)

            for i, parameters in enumerate(parameters_configurations):
                classifier = Classifier(**parameters)

                # accuracy, precision, recall, f-measure
                measures = np.zeros(METRICS_COUNT, dtype=float)

                classification_result = np.zeros(y.shape, dtype=object)

                # predicting class for all the instances
                for train_indexes, test_indexes in folds:
                    x_train, y_train = x[train_indexes], y[train_indexes]
                    x_test, y_test = x[test_indexes], y[test_indexes]
                    classifier.fit(x_train, y_train)
                    probabilities = classifier.predict_proba(x_test)

                    y_pred = classifier.predict(x_test)

                    classification_result[test_indexes] = label_encoder.inverse_transform(y_pred)
                    measures += np.array([
                        accuracy_score(y_test, y_pred),
                        precision_score(y_test, y_pred),
                        recall_score(y_test, y_pred),
                        f1_score(y_test, y_pred),
                    ])
                    # context value = probability of true class "minus" max value of probability of other classes
                    context_column_variants[test_indexes, i] = probabilities[xrange(len(test_indexes)), y_test]
                    probabilities[xrange(len(test_indexes)), y_test] = 0.0
                    context_column_variants[test_indexes, i] -= np.max(probabilities, axis=1)
#                     print "context_column_variants", context_column_variants
                # averaging quality measures
                measures /= FOLDS_NUMBER

                # updating classifier with max f-measure
                if measures[-1] > max_f1:
                    max_f1 = measures[-1]
                    max_i = i

#                 saving results to a file
                readable_params = parameters.copy()
                readable_params.pop('probability', None)
                readable_params.pop('kernel', None)
                if 'n_neighbors' in readable_params:
                    readable_params['k'] = readable_params.pop('n_neighbors', None)
                np.savetxt(
                    '../Loptev_results/{dataset}-{classifier}-{params}.txt'.format(
                        dataset=name,
                        classifier=Classifier.__name__,
                        params='-'.join('{}-{}'.format(k, v) for k, v in readable_params.iteritems())
                    ), classification_result, fmt='%s'
                )

                # printing results

                print ', '.join('{}={}'.format(k, v) for k, v in parameters.iteritems()), measures
            print 'Highest F1:', max_f1, ':', ', '.join('{}={}'.format(k, v)
                                                            for k, v in parameters_configurations[max_i].iteritems())

            context[:, classifier_index] = context_column_variants[:, max_i]
            best_parameters.append(parameters_configurations[max_i])

        classification_result = np.zeros(y.shape, dtype=object)
        print "classification_result", classification_result
        # evaluation of meta classifier
        measures = np.zeros(METRICS_COUNT, dtype=float)

        selected_classifiers = np.zeros(y.shape[0])
        for train_indexes, test_indexes in folds:
            x_train, y_train = x[train_indexes], y[train_indexes]
            x_test, y_test = x[test_indexes], y[test_indexes]

            classifiers = []
            meta_probabilities = np.zeros((len(test_indexes), len(CLASSIFIERS)))
            for i, classifier_and_params in enumerate(CLASSIFIERS):
                Classifier, _ = classifier_and_params
                classifier = Classifier(**best_parameters[i])
                classifier.fit(x_train, y_train)
                classifiers.append(classifier)
                meta_regressor = RandomForestRegressor(300)
                meta_regressor.fit(x_train, context[train_indexes, i])
                meta_probabilities[:, i] = meta_regressor.predict(x_test)

            predicted_classifiers = meta_probabilities.argmax(axis=1)
            selected_classifiers[test_indexes] = predicted_classifiers

            y_pred = np.zeros(len(y_test), dtype=int)
            for i, test_object in enumerate(x_test):
                y_pred[i] = classifiers[predicted_classifiers[i]].predict([test_object])[0]

            classification_result[test_indexes] = label_encoder.inverse_transform(y_pred)

            measures += np.array([
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred),
                recall_score(y_test, y_pred),
                f1_score(y_test, y_pred),
            ])

        measures /= FOLDS_NUMBER

        np.savetxt(
            '../Loptev_results/{dataset}-meta.txt'.format(
                dataset=name,
            ), classification_result, fmt='%s'
        )

        elements, frequencies = np.unique(selected_classifiers, return_counts=True)
        frequencies = frequencies.astype(float)
        frequencies /= frequencies.sum()

