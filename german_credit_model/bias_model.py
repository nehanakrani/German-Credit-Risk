import aif360
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
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
from IPython.display import Markdown, display
from collections import OrderedDict
from aif360.metrics import ClassificationMetric
all_metrics =  ["Statistical parity difference", "Average odds difference"]

def compute_metrics(dataset_true, dataset_pred,
                    unprivileged_groups, privileged_groups,
                    disp = True):
    """ Compute the key metrics """
    classified_metric_pred = ClassificationMetric(dataset_true,
                                                 dataset_pred,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    metrics = OrderedDict()
    metrics["Balanced accuracy"] = 0.5*(classified_metric_pred.true_positive_rate()+
                                             classified_metric_pred.true_negative_rate())
    metrics["Statistical parity difference"] = classified_metric_pred.statistical_parity_difference()
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics["Average odds difference"] = classified_metric_pred.average_odds_difference()
    metrics["Equal opportunity difference"] = classified_metric_pred.equal_opportunity_difference()
    metrics["Theil index"] = classified_metric_pred.theil_index()

    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))

    return metrics


def original_training_dataset():
    df, dict_df= GermanDataset(
        features_to_drop=['personal_status'] # ignore sex-related attributes
    ).convert_to_dataframe()
    german_dataset = GermanDataset(
        protected_attribute_names=['age'],
        privileged_classes=[lambda x: x >= 25],# age >=25 is considered privileged
        features_to_drop=['personal_status', 'sex'] # ignore sex-related attributes
    )
    #dividing the dataset into train and test
    german_dataset_train, german_dataset_test = german_dataset.split([0.7], shuffle=True)
    privileged_groups = [{'age': 1}]
    unprivileged_groups = [{'age': 0}]
    # Get the dataset and split into train and test
    german_dataset_train, german_dataset_vt = german_dataset.split([0.7], shuffle=True)
    german_dataset_valid, german_dataset_test = german_dataset_vt.split([0.5], shuffle=True)
    #Train model
    # Logistic regression classifier and predictions
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(german_dataset_train.features)
    y_train = german_dataset_train.labels.ravel()
    w_train = german_dataset_train.instance_weights.ravel()

    lmod = LogisticRegression()
    lmod.fit(X_train, y_train, sample_weight=german_dataset_train.instance_weights)
    y_train_pred = lmod.predict(X_train)

    # positive class index
    pos_ind = np.where(lmod.classes_ == german_dataset_train.favorable_label)[0][0]
    german_dataset_train_pred = german_dataset_train.copy()
    german_dataset_train_pred.labels = y_train_pred

    pos_ind = np.where(lmod.classes_ == german_dataset_train.favorable_label)[0][0]
    german_dataset_train_pred = german_dataset_train.copy()
    german_dataset_train_pred.labels = y_train_pred
    # Validating the orignal Dataset
    german_dataset_valid_pred = german_dataset_valid.copy(deepcopy=True)
    X_valid = scale_orig.transform(german_dataset_valid_pred.features)
    y_valid = german_dataset_valid_pred.labels
    german_dataset_valid_pred.scores = lmod.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)

    german_dataset_test_pred = german_dataset_test.copy(deepcopy=True)
    X_test = scale_orig.transform(german_dataset_test_pred.features)
    y_test = german_dataset_test_pred.labels
    german_dataset_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)
    num_thresh = 100
    ba_arr = np.zeros(num_thresh)
    class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
    for idx, class_thresh in enumerate(class_thresh_arr):

        fav_inds = german_dataset_valid_pred.scores > class_thresh
        german_dataset_valid_pred.labels[fav_inds] = german_dataset_valid_pred.favorable_label
        german_dataset_valid_pred.labels[~fav_inds] = german_dataset_valid_pred.unfavorable_label

        classified_metric_orig_valid = ClassificationMetric(german_dataset_valid,
                                                german_dataset_valid_pred,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)

        ba_arr[idx] = 0.5*(classified_metric_orig_valid.true_positive_rate()\
                        +classified_metric_orig_valid.true_negative_rate())

    best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    best_class_thresh = class_thresh_arr[best_ind]

    print("Best balanced accuracy (no reweighing) = %.4f" % np.max(ba_arr))
    print("Optimal classification threshold (no reweighing) = %.4f" % best_class_thresh)

    # Predictions from original testing data"
    bal_acc_arr_orig = []
    disp_imp_arr_orig = []
    avg_odds_diff_arr_orig = []

    print("Classification threshold used = %.4f" % best_class_thresh)
    for thresh in tqdm(class_thresh_arr):
        if thresh == best_class_thresh:
            disp = True
        else:
            disp = False

        fav_inds = german_dataset_test_pred.scores > thresh
        german_dataset_test_pred.labels[fav_inds] = german_dataset_test_pred.favorable_label
        german_dataset_test_pred.labels[~fav_inds] = german_dataset_test_pred.unfavorable_label

        metric_test_bef = compute_metrics(german_dataset_test, german_dataset_test_pred,
                                        unprivileged_groups, privileged_groups,
                                        disp = disp)

        bal_acc_arr_orig.append(metric_test_bef["Balanced accuracy"])
        avg_odds_diff_arr_orig.append(metric_test_bef["Average odds difference"])
        disp_imp_arr_orig.append(metric_test_bef["Disparate impact"])

    # x = {'bal_acc_arr_orig' : metric_test_bef["Balanced accuracy"], 'avg_odds_diff_arr_orig' : metric_test_bef["Average odds difference"], 'disp_imp_arr_orig' : metric_test_bef["Disparate impact"], 'best_class_thresh' : np.max(ba_arr), '0ptimal_classification_thr' : best_class_thresh}
    # print(x)
    # return  x



    ##abs(1-disparate impact) must be small (close to 0) for classifier predictions to be fair.
    #However, for a classifier trained with original training data,
    ##at the best classification rate, this is high. This implies unfairness.
    fig, ax1 = plt.subplots(figsize=(10,7))
    ax1.plot(class_thresh_arr, bal_acc_arr_orig)
    ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)

    ax2 = ax1.twinx()
    ax2.plot(class_thresh_arr, np.abs(1.0-np.array(disp_imp_arr_orig)), color='r')
    ax2.set_ylabel('abs(1-disparate impact)', color='r', fontsize=16, fontweight='bold')
    ax2.axvline(best_class_thresh, color='k', linestyle=':')
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)


    # ##average odds difference = 0.5((FPR_unpriv-FPR_priv)+(TPR_unpriv-TPR_priv)) must be close to zero for the classifier to be fair.

    # ##However, for a classifier trained with original training data, at the best classification rate,
    # ##this is quite high. This implies unfairness.
    # fig, ax1 = plt.subplots(figsize=(10,7))
    # ax1.plot(class_thresh_arr, bal_acc_arr_orig)
    # ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
    # ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
    # ax1.xaxis.set_tick_params(labelsize=14)
    # ax1.yaxis.set_tick_params(labelsize=14)


    # ax2 = ax1.twinx()
    # ax2.plot(class_thresh_arr, avg_odds_diff_arr_orig, color='r')
    # ax2.set_ylabel('avg. odds diff.', color='r', fontsize=16, fontweight='bold')
    # ax2.axvline(best_class_thresh, color='k', linestyle=':')
    # ax2.yaxis.set_tick_params(labelsize=14)
    # ax2.grid(True)


original_training_dataset()