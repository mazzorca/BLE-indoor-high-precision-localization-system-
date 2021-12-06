"""
Script to compare the different regressors
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import dataset_generator
import testMultiRegress
import regressors_lib

from sklearn import tree
import graphviz

if __name__ == '__main__':
    performance_test = 3
    choice = 0  # 0 euclidean, 1 squares

    # Test with a dataset with and without the standard deviation and mean
    if performance_test == 0:
        testMultiRegress.main_test_dataset_default_vs_dataset_mean_and_std()

    # Plot a graph for each regressor, in which all the runs are compared with euclidean metric
    if performance_test == 1:
        testMultiRegress.compare_experiment_with_ecdf()

    # Plot a graph for each regressor, in which all the runs are compared with squares metric
    if performance_test == 2:
        testMultiRegress.compare_experiment_with_ecdf_square()

    # cPlot a graph for each run, in which all the regressor are compared
    if performance_test == 3:
        name_files_reader = [["dati3105run0r"], ["dati3105run1r"], ["dati3105run2r"]]
        name_files_cam = [["Cal3105run0"], ["Cal3105run1"], ["Cal3105run2"]]

        # name_files_reader = [["dati3105run0r", "dati3105run1r", "dati3105run2r"]]
        # name_files_cam = [["Cal3105run0", "Cal3105run1", "Cal3105run2"]]

        # x_train, y_train = dataset_generator.generate_dataset_base("BLE2605r", "2605r0")
        # train_dataset = [x_train, y_train]
        train_dataset = dataset_generator.load_dataset_numpy_file("x_train", "y_train")
        num_run = 1
        for name_file_reader, name_file_cam in zip(name_files_reader, name_files_cam):
            # x_test, y_test = dataset_generator.generate_dataset(name_file_reader, name_file_cam,
            #                                                     dataset_generator.generate_dataset_base)

            test_name = dataset_generator.dataset_tests[name_file_reader[0]]
            test_dataset = dataset_generator.load_dataset_numpy_file(test_name[0], test_name[1])
            # test_dataset = [x_test, y_test]
            testMultiRegress.compare_regressor_with_ecdf(train_dataset, test_dataset, name_file_reader, num_run,
                                                         what_type_of_ecdf=choice)
            num_run += 1

    # Plot the different performance of the k-NN varying the k parameters
    if performance_test == 4:
        train_dataset = dataset_generator.load_dataset_numpy_file("x_train_total", "y_train_total")
        # train_dataset = dataset_generator.load_dataset_numpy_file("AllSquare_RSSI", "AllSquare_Target")
        # train_dataset[0] = train_dataset[0].transpose()
        # train_dataset[1] = train_dataset[1].transpose()

        test_dataset = dataset_generator.load_dataset_numpy_file("x_test", "y_test")

        testMultiRegress.compare_k_NNs(2000, 200, train_dataset, test_dataset,
                                       ["dati3105run0r dati3105run1r dati3105run2r"])

    # get the decision tree graph
    if performance_test == 5:
        clf = testMultiRegress.CLASSIFIERS_DICT['Decision Tree']
        train_dataset = dataset_generator.load_dataset_numpy_file("x_train", "y_train")
        x_train = train_dataset[0]
        y_train = train_dataset[1]

        clf = clf.fit(x_train, y_train)
        dot_data = tree.export_graphviz(clf, out_file=None,
                                        filled=True, rounded=True,
                                        special_characters=True)
        graph = graphviz.Source(dot_data)

    # Test splitting the total dataset in 80-20%
    if performance_test == 6:
        set0 = dataset_generator.load_dataset_numpy_file("x_train", "y_train")
        set1 = dataset_generator.load_dataset_numpy_file("x_test0", "y_test0")
        set2 = dataset_generator.load_dataset_numpy_file("x_test1", "y_test1")
        set3 = dataset_generator.load_dataset_numpy_file("x_test2", "y_test2")

        total_dataset_x = np.concatenate([set0[0], set1[0], set2[0], set3[0]])
        total_dataset_y = np.concatenate([set0[1], set1[1], set2[1], set3[1]])

        x_train, x_test, y_train, y_test = train_test_split(total_dataset_x, total_dataset_y, train_size=0.05)

        train_dataset = [x_train, y_train]
        test_dataset = [x_test, y_test]

        testMultiRegress.compare_regressor_with_ecdf(train_dataset, test_dataset, ["All Dataset split"],
                                                     what_type_of_ecdf=choice)

    # X, y = utility.load_dataset_arff("datasets/arff/datasetTrain0")
    # X, y = dataset_generator.generate_dataset_base_all()
    # x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.70)
    # plot_test_multi_regress(x_train, x_test, y_train, y_test)
    #
    # X, y = dataset_generator.generate_dataset_with_mean_and_std_all()
    # x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.70)
    # plot_test_multi_regress(x_train, x_test, y_train, y_test, title_add="mean_std")
