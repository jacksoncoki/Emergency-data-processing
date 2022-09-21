import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class MissingImpute:
    def __init__(self, original_data):
        self.original_data = original_data
        self.complete_dataset()
        self.impute_RF()
        
    def impute_RF(self):
        '''
        For each feature, the non missing samples are used as 
        the training set, and the missing samples are used as 
        the test set to train the random forest model for interpolation
        :return: Complete data set of model interpolation
        '''
        self.RF_impute_data = self.original_data.copy()
        for col in self.RF_impute_data.columns:
            if not len(self.RF_impute_data[self.RF_impute_data[col].isna()]):
                continue
            train_index = self.RF_impute_data[~self.RF_impute_data[col].isna()].index
            test_index = self.RF_impute_data[self.RF_impute_data[col].isna()].index
            train_data = self.complete_data.loc[train_index, :]
            test_data = self.complete_data.loc[test_index, :]
            X_train = train_data.drop(columns=[col])
            y_train = train_data[col]

            X_test = test_data.drop(columns=[col])
            if self.RF_impute_data[col].dtype == object:
                self.rf = RandomForestClassifier(n_estimators=60, max_depth=16, min_samples_split=6)
                self.rf.fit(X_train, y_train)
                pre = self.rf.predict(X_test)
            else: 
                self.rf = RandomForestRegressor(n_estimators=60, max_depth=16, min_samples_split=6)
                self.rf.fit(X_train, y_train)
                pre = self.rf.predict(self.x_test)
            impute_value = pd.Series(index=test_index, data=pre)
            self.RF_impute_data.loc[impute_value.index, col] = impute_value
        return self.RF_impute_data

    def complete_dataset(self):
            '''
            A complete data set is generated according to the original data, and the missing data are interpolated by
            means (continuous variables) and mode (discrete variables)
            :return:
            '''
            self.complete_data = self.original_data.copy()
            for col in self.original_data.columns:
                if self.original_data[col].dtype == object:
                    impute_value = self.original_data[col].mode()
                else:
                    impute_value = self.original_data[col].mean()
                mis_f = self.original_data[self.original_data[col].isna()][col]
                if not len(mis_f):
                    continue
                self.complete_data.loc[mis_f.index, col] = impute_value
        

    

