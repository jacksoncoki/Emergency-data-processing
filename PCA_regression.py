from statsmodels.api import Logit
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.utils import resample
import numpy as np
from scipy.stats import ttest_1samp


def pca_regression(samples):
    '''
    Perform 500 fold bootstrapping for model
    training and internal verification of principal
    component regression. Only the content of model
    training is reserved here to simplify the code. The calculation
    of model performance indicators is not shown here
    :param samples: Sparse sample data for analysis
    :return:prediction result
    '''
    
    for i in range(500):
        train = resample(samples, n_samples=int(len(samples) * 0.7), replace=False)
        test_index = pd.Index(set(samples.index) - set(train.index))
        test = samples.loc[test_index]
        pca = PCA(n_components=17, copy=True)
        new_train_set = pca.fit_transform(train.iloc[:, :-1])
        new_test_set = pca.transform(test.iloc[:, :-1])
        lr = Logit(train.iloc[:, -1], new_train_set)
        result = lr.fit(method='newton', maxiter=100, disp=False)
        pre = list(map(round, result.predict(new_test_set)))



