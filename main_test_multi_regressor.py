import dataset_generator
import testMultiRegress

if __name__ == '__main__':
    performance_test = 3
    choice = 0

    if performance_test == 0:
        testMultiRegress.main_test_dataset_default_vs_dataset_mean_and_std()

    if performance_test == 1:
        testMultiRegress.compare_experiment_with_ecdf()

    if performance_test == 2:
        testMultiRegress.compare_experiment_with_ecdf_square()

    if performance_test == 3:
        name_files_reader = [["dati3105run0r"], ["dati3105run1r"], ["dati3105run2r"]]
        name_files_cam = [["Cal3105run0"], ["Cal3105run1"], ["Cal3105run2"]]

        # name_files_reader = [["dati3105run0r", "dati3105run1r", "dati3105run2r"]]
        # name_files_cam = [["Cal3105run0", "Cal3105run1", "Cal3105run2"]]

        x_train, y_train = dataset_generator.generate_dataset_base("BLE2605r", "2605r0")
        train_dataset = [x_train, y_train]
        for name_file_reader, name_file_cam in zip(name_files_reader, name_files_cam):
            x_test, y_test = dataset_generator.generate_dataset(name_file_reader, name_file_cam,
                                                                dataset_generator.generate_dataset_base)

            test_dataset = [x_test, y_test]
            testMultiRegress.compare_regressor_with_ecdf(train_dataset, test_dataset, name_file_reader,
                                                         what_type_of_ecdf=choice)

    if performance_test == 4:
        train_dataset = dataset_generator.load_dataset_numpy_file("x_train", "y_train")
        test_dataset = dataset_generator.load_dataset_numpy_file("x_test2", "y_test2")

        testMultiRegress.compare_k_NNs(1000, 100, train_dataset, test_dataset,
                                       "dati3105run2r")

    # X, y = utility.load_dataset_arff("datasets/arff/datasetTrain0")
    # X, y = dataset_generator.generate_dataset_base_all()
    # x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.70)
    # plot_test_multi_regress(x_train, x_test, y_train, y_test)
    #
    # X, y = dataset_generator.generate_dataset_with_mean_and_std_all()
    # x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.70)
    # plot_test_multi_regress(x_train, x_test, y_train, y_test, title_add="mean_std")
