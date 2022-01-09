"""Provides end-to-end flow to use LightGBM model.
"""
__author__ = 'khanhtpd'
__date__ = '2022-01-01'

from typing import Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import optuna

# from optuna.visualization import plot_contour
# from optuna.visualization import plot_edf
# from optuna.visualization import plot_intermediate_values
# from optuna.visualization import plot_optimization_history
# from optuna.visualization import plot_parallel_coordinate
# from optuna.visualization import plot_param_importances
# from optuna.visualization import plot_slice


def accuracy_macro(pred: list, true: list) -> float:
    """Calculate macro-accuracy.

    Args:
        pred (list): Prediction. Values should be either 0 or 1.
        true (list): True. Values should be either 0 or 1.

    Returns:
        float: macro-accuracy
    """
    accuracy_male = [1 if (x == 1) & (y == 1) else 0 for (x, y) in zip(pred, true)]
    # plus epsilon in case denominator = 0
    accuracy_male = sum(accuracy_male) / (sum(pred) + 1.e-16)

    accuracy_female = [1 if (x == 0) & (y == 0) else 0 for (x, y) in zip(pred, true)]
    # plus epsilon in case denominator = 0
    accuracy_female = sum(accuracy_female) / (sum([1 - x for x in pred]) + 1.e-16)

    accuracy = (accuracy_male + accuracy_female) / 2
    return accuracy


class MLFramework:

    def __init__(self):
        self.data: pd.DataFrame = None
        self.train: lgb.Dataset = None
        self.valid: lgb.Dataset = None
        self.test: lgb.Dataset = None
        self.study: optuna.study = None
        self.best_params: dict = None
        self.booster: lgb.Booster = None
        self.cvbooster: lgb.CVBooster = None

    def train_valid_test_split(
        self,
        label_col: Union[str, pd.Series],
        feat_cols: Union[str, pd.DataFrame],
        data: pd.DataFrame = None,
        valid_size: Union[int, float] = 0.0,
        test_size: Union[int, float] = 0.2,
        label_time_col: str = None,
        categorical_col: Union[list, str] = 'auto'
    ) -> Tuple[lgb.Dataset, lgb.Dataset, lgb.Dataset]:
        """Split data to train, validation and test sets.

        Args:
            label_col (Union[str, pd.Series]): Label.
            feat_cols (Union[str, pd.DataFrame]): Features.
            data (pd.DataFrame, optional): Input data frame. Should include both label and features.
                Defaults to None.
            valid_size (Union[int, float], optional): If int, number of ending periods of
                label_time_col to be in validation set. Arrangement: train -> validation -> test.
                If float, proportion of validation set. Defaults to 0.
            test_size (Union[int, float], optional): If int, number of ending periods of
                label_time_col to be in test set. Arrangement: train -> validation -> test.
                If float, proportion of test set. Defaults to 0.2.
            label_time_col (str, optional): Label time column name. Need to specify this in case
                val_size and test_size are integers. Defaults to None.
            categorical_col (Union[list, str], optional): List of categorical features.
                Defaults to 'auto'.

        Returns:
            Tuple[lgb.Dataset, lgb.Dataset, lgb.Dataset]: train, valid and test lgb dataset
        """
        # if whole data is not provided, i.e. label_col is pd.Series and feat_cols is pd.DataFrame
        if data is None:
            data = feat_cols.assign(**{'LABEL': label_col})
            label_col = 'LABEL'
            feat_cols = feat_cols.columns

        self.data = data

        # if sizes are proportion
        if (valid_size < 1) & (test_size < 1):
            test = data.sample(frac=test_size, random_state=0)
            valid = (
                data[~data.index.isin(test.index)]
                .sample(frac=valid_size / (1 - test_size), random_state=0)
            )
            train = data[~data.index.isin(list(test.index) + list(valid.index))]

        # if sizes are number of ending periods
        else:
            if label_time_col is None:
                raise ValueError('label_time_col must be specified.')
            label_time = data[label_time_col].unique()
            label_time = np.sort(label_time)

            label_time_test = label_time[-test_size:]
            label_time_valid = label_time[-(test_size + valid_size):-test_size]

            test = data[data[label_time_col].isin(list(label_time_test))]
            valid = data[data[label_time_col].isin(list(label_time_valid))]
            train = data[~data[label_time_col].isin(list(label_time_test) + list(label_time_valid))]

        x_train, x_valid, x_test = train[feat_cols], valid[feat_cols], test[feat_cols]
        y_train, y_valid, y_test = train[label_col], valid[label_col], test[label_col]

        train = lgb.Dataset(data=x_train, label=y_train, categorical_feature=categorical_col, free_raw_data=False)
        valid = lgb.Dataset(data=x_valid, label=y_valid, reference=train, free_raw_data=False)
        test = lgb.Dataset(data=x_test, label=y_test, categorical_feature=categorical_col, free_raw_data=False)
        self.train, self.valid, self.test = train, valid, test
        print('Data and its partition (train, valid, test) are stored in attributes data, train, ', end='')
        print('valid and test, respectively.')

    def accuracy_macro_lgb(pred, data):
        true = data.get_label()
        pred = np.where(pred < 0.5, 0, 1)
        accuracy = accuracy_macro(pred, true)
        return 'accuracy_macro', accuracy, True

    def _objective_lgb(trial, train: lgb.Dataset) -> float:
        """Define objective function for optuna search.

        Args:
            train (lgb.Dataset): Training data set.
            valid (lgb.Dataset): Validation data set. For early stopping and validate score purpose.

        Returns:
            float: AUC score on validation data set.
        """
        params = {
            'objective': 'binary',  # binary log loss classification
            'metric': 'custom',  # metric to be evaluated on the evaluation set for early stopping
            'learning_rate': 0.1,  # should not be turned
            'boosting_type': 'gbdt',  # early stopping is not available in dart mode
            'verbose': -1,  # suppress warning
            'feature_pre_filter': False,  # to be able tunning min_child_samples

            'num_leaves': trial.suggest_int('num_leaves', 15, 1023),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_data_in_leaf', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.3, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
        }
        gbm = lgb.cv(
            params=params,
            train_set=train,
            num_boost_round=500,
            early_stopping_rounds=50,
            verbose_eval=False,
            feval=MLFramework.accuracy_macro_lgb
        )
        accuracy = max(gbm['accuracy_macro-mean'])
        return accuracy

    def optuna_lgb(self, n_trials=100):
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: MLFramework._objective_lgb(trial, self.train),
            n_trials=n_trials
        )
        self.study = study
        self.best_params = {
            'objective': 'binary',
            'metric': 'custom',
            'learning_rate': 0.1,
            'boosting_type': 'gbdt',
            'verbose': -1,
            'feature_pre_filter': False
        }
        self.best_params.update(study.best_params)
        self.cvbooster = lgb.cv(
            params=self.best_params,
            train_set=self.train,
            num_boost_round=500,
            early_stopping_rounds=50,
            verbose_eval=False,
            feval=MLFramework.accuracy_macro_lgb,
            return_cvbooster=True
        )
        self.booster = lgb.train(
            params=self.best_params,
            train_set=self.train,
            num_boost_round=len(self.cvbooster['accuracy_macro-mean'])
        )
        print(f"Best booster has been trained with num_boost_round={len(self.cvbooster['accuracy_macro-mean'])}.")
        print('New attributes assigned: study, best_params, cvbooster, booster.')

    # plot_contour(study)
    # plot_edf(study)
    # plot_intermediate_values(study)
    # plot_optimization_history(study)
    # plot_optimization_history(study)
    # plot_parallel_coordinate(study)
    # plot_param_importances(study)
    # plot_slice(study)

    def save_model(self, file_path: str):
        """Save lgb booster.

        Args:
            file_path (str): Location and file name of the booster.
        """
        self.booster.save_model(file_path)

    def load_model(self, file_path: str):
        """Load lgb booster.

        Args:
            file_path (str): Location and file name of the booster.
        """
        self.booster = lgb.Booster(model_file=file_path)

    def get_feature_importance(
        self,
        ntop: int = 10,
        importance_type: str = 'gain'
    ) -> pd.Series:
        """Get top feature importance from trained booster.

        Args:
            n (int, optional): Number of top features. Defaults to 10.
            importance_type (str, optional): 'gain' or 'split'. Defaults to 'gain'.

        Returns:
            pd.Series: Top feature importance.
        """
        importance = pd.Series(
            data=self.booster.feature_importance(importance_type=importance_type),
            index=self.booster.feature_name(),
            name='importance'
        )
        importance = importance.sort_values(ascending=False).head(ntop)
        return importance

    def plot_feature_importance(
        self,
        ntop: int = 10,
        importance_type: str = 'gain',
        fmt: str = '%.0f',
        padding: int = 5
    ):
        """Plot feature importance from trained booster.

        Args:
            ntop (int, optional): Number of top features. Defaults to 10.
            importance_type (str, optional): 'gain' or 'split'. Defaults to 'gain'.
            fmt (str, optional): Format of printing values. Defaults to '%.0f'.
            padding (int, optional): Space between values and bars. Defaults to 5.
        """
        importance = self.get_feature_importance(ntop, importance_type)
        ax = sns.barplot(x=importance, y=importance.index)
        plt.xlabel('')
        plt.title('Feature Importance', size=14)
        ax.bar_label(ax.containers[0], fmt=fmt, padding=padding)

    def predict(self, x_data: pd.DataFrame, cv: bool = True) -> np.array:
        """Predict propensity score using lgb booster.

        Args:
            x_data (pd.DataFrame): Feature data set.
            cv (bool): If using (aggregate) cross validation boosters. Defaults to True.

        Returns:
            np.array: Propensity score.
        """
        if cv:
            y_pred = self.cvbooster['cvbooster'].predict(x_data)
            y_pred = np.mean(y_pred, axis=0)
        else:
            y_pred = self.booster.predict(x_data)
        return y_pred


def get_gain_table(y_true: list, y_pred: list, n_level: int = 10) -> pd.DataFrame:
    """Get gain table.

    Args:
        y_true (list): True label.
        y_pred (list): Prediction score.
        n_level (int, optional): Number of levels. Defaults to 10.

    Returns:
        pd.DataFrame: Gain table.
    """
    label_levels = reversed(range(1, 1 + n_level))
    label_levels = ['Level' + str(x) for x in label_levels]
    level = pd.qcut(y_pred, n_level, labels=label_levels, duplicates='raise')
    gain_df = pd.DataFrame({'true': y_true, 'predict': y_pred, 'level': level})
    gain_df = (
        gain_df
        .groupby('level')
        .agg(
            predict_min=('predict', 'min'),
            predict_mean=('predict', 'mean'),
            predict_max=('predict', 'max'),
            true_count=('true', 'count'),
            true_sum=('true', 'sum'),)
        .assign(true_mean=lambda df: df['true_sum'] / df['true_count'])
        .sort_index(ascending=False)
        .reset_index()
    )
    return gain_df


def plot_calibration_curve(y_true: list, y_pred: list, n_level: int = 10):
    """Plot calibration curve

    Args:
        y_true (list): True label.
        y_pred (list): Prediction score.
        n_level (int, optional): Number of levels. Defaults to 10.
    """
    gain_df = get_gain_table(y_true, y_pred, n_level)
    sns.lineplot(x='predict_mean', y='true_mean', data=gain_df, marker='o')
    max_predict_mean = max(gain_df['predict_mean'])
    max_true_mean = max(gain_df['true_mean'])
    lim_axis = max(max_predict_mean, max_true_mean)
    plt.plot([0.0, lim_axis], [0.0, lim_axis], linestyle='-.', color='grey', linewidth=0.5)
    plt.title('Calibration Curve', size=14)
