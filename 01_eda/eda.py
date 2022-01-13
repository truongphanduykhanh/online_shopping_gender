"""Provides common functions to Exploratory Data Analysis.
"""
__author__ = 'khanhtpd'
__date__ = '2022-01-01'


import math
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def count_missing_columns_rows(data: pd.DataFrame, figsize: tuple = (12, 4)):
    """Count missing values according to columns and rows.

    Args:
        data (pd.DataFrame): Input data frame.
        figsize (tuple, optional): Size of figure. Defaults to (12, 4).
    """
    bins = np.linspace(0, 1, 11)
    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    missing_column = data.isna().sum(axis=0) / data.shape[0]
    missing_column = (
        pd.cut(missing_column, bins=bins, include_lowest=False, right=True)
        .cat.add_categories(pd.Interval(0.0, 0.0, closed='both')))
    missing_column.cat.reorder_categories(missing_column.cat.categories.sortlevel()[0], inplace=True)
    missing_column = (
        missing_column
        .fillna(pd.Interval(0, 0, closed='both'))
        .value_counts()
        .sort_index()
    )
    ax = missing_column.plot(kind='barh')
    total = missing_column.sum()
    for p in ax.patches:
        percentage = 100 * p.get_width() / total
        percentage = f'{percentage:.1f}%'
        x = p.get_x() + p.get_width()
        y = p.get_y() + 0.05
        ax.annotate(percentage, (x, y))
    plt.xlabel('Count')
    plt.ylabel('Percentage of missing')
    plt.title('Missing values - Columns', size=15)

    plt.subplots_adjust(wspace=0.5)
    plt.subplot(1, 2, 2)
    missing_row = data.isna().sum(axis=1) / data.shape[1]
    missing_row = (
        pd.cut(missing_row, bins=bins, include_lowest=False, right=True)
        .cat.add_categories(pd.Interval(0.0, 0.0, closed='both')))
    missing_row.cat.reorder_categories(missing_row.cat.categories.sortlevel()[0], inplace=True)
    missing_row = (
        missing_row
        .fillna(pd.Interval(0, 0, closed='both'))
        .value_counts()
        .sort_index()
    )
    ax = missing_row.plot(kind='barh')
    total = missing_row.sum()
    for p in ax.patches:
        percentage = 100 * p.get_width() / total
        percentage = f'{percentage:.1f}%'
        x = p.get_x() + p.get_width()
        y = p.get_y() + 0.05
        ax.annotate(percentage, (x, y))
    plt.xlabel('Count')
    plt.ylabel('Percentage of missing')
    plt.title('Missing values - Rows', size=15)


def top_missing_columns(data: pd.DataFrame, ntop: int = 10, figsize: Union[tuple, str] = 'auto'):
    """Count missing values in every columns.

    Args:
        data (pd.DataFrame): Input data frame.
        ntop (int, optional): Number of top missing columns displayed. Defaults to 10.
        figsize (Union[tuple, str], optional): Size of the whole plot. If 'auto',
            figsize = (12, ntop / 2). Defaults to 'auto'.
    """
    if ntop == 'all':
        ntop = data.shape[1]
    if figsize == 'auto':
        figsize = (12, ntop / 2)
    plt.figure(figsize=figsize)
    missing_count = data.isna().sum()
    missing_count = missing_count.sort_values(ascending=False)[0: ntop]
    missing_count = missing_count.sort_values(ascending=True)
    ax = missing_count.plot(kind='barh')
    for p in ax.patches:
        percentage = p.get_width() / len(data) * 100
        percentage = f'{percentage:.1f}%'
        x = p.get_x() + p.get_width()
        y = p.get_y() + 0.15
        ax.annotate(percentage, (x, y))
    plt.title('Top missing values columns', size=15)


def displot(
    data: pd.DataFrame,
    columns: list = None,
    kind: str = 'hist',
    nrows: Union[int, str] = 'auto',
    ncols: int = 2,
    figsize: Union[tuple, str] = 'auto',
    hspace: float = 0.7,
    wspace: float = 0.5,
    title: str = 'Distribution of numerical variables',
    y_title: float = 1
):
    """Plot distribution plot of continuous variables.

    Args:
        data (pd.DataFrame): Input data frame.
        columns (list, optional): Names of numerical columns in the data frame.
            If None, numerical columns will be taken. Defaults to None.
        kind (str, optional): Kind of plot. Defaults to 'hist'.
        nrows (Union[int, str], optional): Number of rows in the plot.
            If 'auto', will be automatically calulated based on ncols. Defaults to 'auto'.
        ncols (int, optional): Number of columns in the plot. Defaults to 2.
        figsize (Union[tuple, str], optional): Size of the whole plot. If 'auto',
            figsize = (12, 2 * nrows). Defaults to 'auto'.
        hspace (float, optional): Height space between sup plots. Defaults to 0.7.
        wspace (float, optional): Width space between sup plots. Defaults to 0.5.
        title (str, optional): Title. Defaults to 'Distribution of numerical variables'.
        y_title (float, optional): Position of title. Defaults to 1.
    """
    if columns is None:
        columns = data.select_dtypes(include=np.number).columns
    if len(columns) == 1:
        ncols = 1
    if nrows == 'auto':
        nrows = math.ceil(len(columns) / ncols)
    if figsize == 'auto':
        figsize = (12, 2 * nrows)
    plt.figure(figsize=figsize)
    for i, column in enumerate(columns):
        plt.subplot(nrows, ncols, i + 1)
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        data[column].plot(kind=kind)
        plt.xlabel('')
        plt.ylabel('')
        plt.title(column)
    if len(columns) > 1:
        plt.suptitle(title, size=15, y=y_title)


def boxplot(
    data: pd.DataFrame,
    columns: list = None,
    label: str = None,
    nrows: Union[int, str] = 'auto',
    ncols: int = 2,
    figsize: Union[tuple, str] = 'auto',
    hspace: float = 0.7,
    wspace: float = 0.5,
    y_title: float = 1,
    title: str = 'Distribution of numerical variables',
):
    """Plot boxplot of numerical variables.

    Args:
        data (pd.DataFrame): Input data frame.
        columns (list, optional): Names of numerical columns in the data frame.
            If None, numerical columns will be taken. Defaults to None.
        label (str, optional): Name of column label in the data frame. Defaults to None.
        nrows (Union[int, str], optional): Number of rows in the plot.
            If 'auto', will be automatically calulated based on ncols. Defaults to 'auto'.
        ncols (int, optional): Number of columns in the plot. Defaults to 2.
        figsize (Union[tuple, str], optional): Size of the whole plot. If 'auto',
            figsize = (12, 2 * nrows). Defaults to 'auto'.
        hspace (float, optional): Height space between sup plots. Defaults to 0.7.
        wspace (float, optional): Width space between sup plots. Defaults to 0.5.
        title (str, optional): Title. Defaults to 'Distribution of numerical variables'.
        y_title (float, optional): Position of title. Defaults to 1.
    """
    if columns is None:
        columns = data.select_dtypes(include=np.number).columns
    if len(columns) == 1:
        ncols = 1
    if nrows == 'auto':
        nrows = math.ceil(len(columns) / ncols)
    if figsize == 'auto':
        figsize = (12, 2 * nrows)
    plt.figure(figsize=figsize)
    for i, column in enumerate(columns):
        plt.subplot(nrows, ncols, i + 1)
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        sns.boxplot(x=column, y=label, data=data, orient='h')
        plt.xlabel('')
        plt.ylabel('')
        plt.title(column)
    if len(columns) > 1:
        plt.suptitle(title, size=15, y=y_title)


def countplot(
    data: pd.DataFrame,
    columns: list = None,
    label: str = None,
    nclass: Union[int, str] = 5,
    nrows: Union[int, str] = 'auto',
    ncols: int = 2,
    figsize: Union[tuple, str] = 'auto',
    sort_index: bool = False,
    sample: int = 10**6,
    hspace: float = 0.7,
    wspace: float = 0.5,
    title: str = 'Distribution of categorical variables',
    y_title: float = 1,
):
    """Count plot categorical variables.

    Args:
        data (pd.DataFrame): Input data frame.
        columns (list, optional): Name of categorical columns in the data frame. If None,
            categorical and string columns will be taken. Defaults to None.
        label (str, optional):  Name of label column in the data frame. Defaults to None.
        nclass (Union[int, str], optional): Number of class displayed in the plot.
            If 'all', display all classes. Defaults to 5.
        nrows (Union[int, str], optional): Number of rows in the plot.
            If 'auto', will be automatically calulated based on ncols. Defaults to 'auto'.
        ncols (int, optional): Number of columns in the plot. Defaults to 2.
        figsize (Union[tuple, str], optional): Size of the whole plot. If 'auto',
            figsize = (12, 2 * nrows). Defaults to 'auto'.
        sort_index (bool, optional): Sort by index. Defaults to False.
        sample (int, optional): Number of drown samples if the dataset is too large.
            Defaults to 10**6.
        hspace (float, optional): Height space between sup plots. Defaults to 0.7.
        wspace (float, optional): Width space between sup plots. Defaults to 0.5.
        title (str, optional): Title. Defaults to 'Distribution of categorical variables'.
        y_title (float, optional): Position of title. Defaults to 1.
    """
    if columns is None:
        columns = data.select_dtypes(exclude=np.number).columns
    if len(data) > sample:
        print(f'Only {sample:,} random samples out of {len(data):,} are taken.')
        data = data.sample(sample, random_state=0)
    if nclass == 'all':
        nclass = data[columns].nunique().max()
    if len(columns) == 1:
        ncols = 1
    if nrows == 'auto':
        nrows = math.ceil(len(columns) / ncols)
    if figsize == 'auto':
        figsize = (12, 2 * nrows)

    plt.figure(figsize=figsize)
    for i, column in enumerate(columns):
        plt.subplot(nrows, ncols, i + 1)
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        # config sort_index
        if sort_index:
            order = (
                data[column]
                .value_counts()
                .iloc[0: min(data[column].nunique(), nclass)]
                .sort_index(ascending=False)
                .index)
        else:
            order = (
                data[column]
                .value_counts()
                .iloc[0: min(data[column].nunique(), nclass)]
                .index)
        # config plot
        if column == label:
            ax = sns.countplot(
                y=column,
                data=data,
                order=order)
        else:
            ax = sns.countplot(
                y=column,
                data=data,
                order=order,
                hue=label)
        # add percentage to the plot
        total = data.shape[0]
        for p in ax.patches:
            percentage = 100 * p.get_width() / total
            percentage = f'{percentage:.1f}%'
            x = p.get_x() + p.get_width()
            if label is None:
                y_adjust = 0.55
            else:
                y_adjust = 0.35
            y = p.get_y() + y_adjust
            ax.annotate(percentage, (x, y))
        # label and title
        plt.title(f'{column} ({data[column].nunique()})', size=12)
        plt.xlabel('')
        plt.ylabel('')

    if len(columns) > 1:
        plt.suptitle(title, size=15, y=y_title)


def correlation_matrix(data: pd.DataFrame, figsize: tuple = (7, 7)):
    """Plot correlation matrix.

    Args:
        data (pd.DataFrame): Input data frame.
        figsize (tuple, optional): Size of the figure. Defaults to (7, 7).
    """
    # Compute the correlation matrix
    corr = data.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(7, 7))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Correlation among variables', size=15)


def categorical_to_numeric(
    data: pd.DataFrame,
    features: Union[str, list],
    label: str = None,
    method: str = 'target',
    drop_first: bool = False,
    missing_value_integer: Union[int, str] = -1
) -> pd.DataFrame:
    """Convert categorical to numeric variables.

    Args:
        data (pd.DataFrame): Input data frame.
        features (Union[str, list]): Names of categorical features.
        label (str, optional): Name of label column, must be specified if method='label'.
            Defaults to None.
        method (str, optional): Method to convert categorical to numeric. Defaults to 'target'.
        drop_first (bool, optional): If method='one_hot', option to drop the first converted
            one-hot column. Defaults to False.
        missing_value_integer (Union[int, str], optional): If method='integer', option to convert
            missing values to maximum number of classes, np.nan or -1. Defaults to -1.

    Raises:
        ValueError: Choose method in ['target', 'one_hot', 'integer'].

    Returns:
        pd.DataFrame: Converted data frame.
    """
    if method not in ['target', 'one_hot', 'integer']:
        raise ValueError("method must be one of ['target', 'one_hot', 'integer'].")
    converted_data = data.copy()

    if type(features) is not list:  # option to input only one feature
        features = [features]

    if method == 'target':  # missing values remain missing values.
        for feature in features:
            feature_label = (
                converted_data
                .groupby(feature)
                .agg({label: np.mean})
                .reset_index())
            feature_label_dict = dict(zip(feature_label[feature], feature_label[label]))
            converted_data[feature] = converted_data[feature].replace(feature_label_dict)
        print(f'Missing values, if any, remain as missing values.')

    elif method == 'one_hot':  # missing values have 0 for all one-hot columns.
        for feature in features:
            dummies = pd.get_dummies(converted_data[feature], drop_first=drop_first)
            data_drop = converted_data.drop(feature, axis=1)
            converted_data = pd.concat([data_drop, dummies], axis=1)
        print(f'Missing values, if any, have 0 for all the converted one-hot columns.')

    elif method == 'integer':  # missing values are encoded as predetermined values.
        global categories_dict_list
        categories_dict_list = []
        for feature in features:
            if missing_value_integer == 'max':
                missing_value_integer = converted_data[feature].astype('category').cat.codes.max() + 1
            categories = converted_data[feature].astype('category').cat.categories
            categories_dict = dict(zip(categories, range(len(categories))))
            categories_dict_list.append(categories_dict)
            converted_data[feature] = (
                converted_data[feature]
                .astype('category').cat.codes.replace(-1, missing_value_integer))
        print(f'Categorical(s) have been encoded according to categories_dict_list.')
        print(f'Missing values, if any, are encoded as maximum classes, np.nan or -1 (defaulted).')
    return converted_data
