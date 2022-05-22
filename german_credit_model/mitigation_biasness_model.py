# Load all necessary packages
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
import matplotlib.pyplot as plt
all_metrics =  ["Statistical parity difference", "Average odds difference"]
# loading the dataset, Dividing dataset into train and test dataset and Display datset shape
# NOTE: #This dataset also includes a protected feature attribute  "sex" which is not consider in this evaluation

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
y_train_pred = lmod.predict(X_train)
