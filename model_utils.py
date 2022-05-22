import aif360
import sys
import numpy as np
from tqdm import tqdm
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
all_metrics =  ["Statistical parity difference", "Average odds difference"]


def lr_bias_model(x_test_data):
    dataset_orig = GermanDataset(
        protected_attribute_names=['age'],
        privileged_classes=[lambda x: x >= 25],# age >=25 is considered privileged
        features_to_drop=['personal_status', 'sex'] # ignore sex-related attributes
    )
    #dividing the dataset into train and test
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
    privileged_groups = [{'age': 1}]
    unprivileged_groups = [{'age': 0}]
    # Get the dataset and split into train and test
    dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)
    #Train model
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()
    w_train = dataset_orig_train.instance_weights.ravel()
    lmod = LogisticRegression()
    lmod.fit(X_train, y_train, sample_weight=dataset_orig_train.instance_weights)
    #test model
    X_test = scale_orig.transform([x_test_data])
    predict = lmod.predict(X_test)
    print("_______________PRED_______", predict)
    return predict

def lr_bias_mitigated_model(x_test_data):
    german_dataset = GermanDataset(protected_attribute_names=['age'],
        privileged_classes=[lambda x: x >= 25], # age >=25 is considered privileged
        features_to_drop=['personal_status', 'sex'] # ignore sex-related attributes
    )
    german_dataset_train, german_dataset_test = german_dataset.split([0.7], shuffle=True)
    privileged_groups = [{'age': 1}]
    unprivileged_groups = [{'age': 0}]
    german_dataset_train, german_dataset_vt = german_dataset.split([0.7], shuffle=True)
    german_dataset_valid, german_dataset_test = german_dataset_vt.split([0.5], shuffle=True)
    # Metric for the original dataset:
    metric_orig_train = BinaryLabelDatasetMetric(german_dataset_train,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)
    print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
    # Mitigation of Biasness
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
    RW.fit(german_dataset_train)
    dataset_transf_train = RW.transform(german_dataset_train)
    # Transformed training dataset
    metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train,
                                            unprivileged_groups=unprivileged_groups,
                                            privileged_groups=privileged_groups)
    print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())
    # Applying Logistics Regresssion on Transformed Data(Bias Mitigated Dataset)
    scale_transf = StandardScaler()
    X_train = scale_transf.fit_transform(dataset_transf_train.features)
    y_train = dataset_transf_train.labels.ravel()
    lmod = LogisticRegression()
    lmod.fit(X_train, y_train,
            sample_weight=dataset_transf_train.instance_weights)
    #test model
    X_test = scale_transf.transform([x_test_data])
    predict = lmod.predict(X_test)
    print("_______________PRED_______", predict)
    return predict
