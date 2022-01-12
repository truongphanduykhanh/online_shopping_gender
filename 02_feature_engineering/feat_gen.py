from datetime import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class ShoppingFeatureEngineering:

    def __init__(self):
        self.raw = None
        self.label = None
        self.feature = None
        self.data = None

    def load_raw(self, feature_path: str, label_path: str):
        feature = pd.read_csv(feature_path, names=['sessionid', 'start', 'end', 'productid'])
        label = pd.read_csv(label_path, names=['gender'])
        raw = feature.assign(gender=label['gender'])
        self.raw = raw

    def encode_label(self):
        gender_dict = {'male': 1, 'female': 0}
        self.raw = self.raw.assign(male=lambda df: df['gender'].replace(gender_dict))
        self.label = self.raw.set_index('sessionid')['male']

    def gen_feat_time(self):
        feat_time = (
            self.raw
            .set_index('sessionid')
            .loc[:, ['start', 'end']]
            .assign(start=lambda df: pd.to_datetime(df['start']))
            .assign(end=lambda df: pd.to_datetime(df['end']))

            .assign(duration=lambda df: (df['end'] - df['start']).dt.total_seconds())
            .assign(dayname=lambda df: df.start.dt.day_name())
            .assign(weekend=lambda df: df['dayname'].isin(['Saturday', 'Sunday']))
            .assign(worktime=lambda df: df['start'].dt.time.between(time(9, 0, 0), time(17, 0, 0)))
            .assign(worktime=lambda df: (df['weekend'] == 0) & (df['worktime'] == 1))
            .assign(nighttime=lambda df: df['start'].dt.hour.isin([22, 23, 0, 1, 2, 3, 4]))
            .assign(oclock=lambda df: df['start'].dt.minute.isin([55, 56, 57, 58, 59, 0, 1, 2, 3, 4]))

            .drop(['start', 'end'], axis=1)
        )
        feat_time['dayname'] = feat_time['dayname'].astype('category')
        feat_time[['duration', 'weekend', 'worktime', 'nighttime', 'oclock']] = (
            feat_time[['duration', 'weekend', 'worktime', 'nighttime', 'oclock']].astype(int))

        self.feature = pd.concat([self.feature, feat_time], axis=1)

    def transform_product(data: pd.DataFrame) -> pd.DataFrame:
        """Separate four levels of productid into different columns.

        Args:
            data (pd.DataFrame): Data input.

        Returns:
            pd.DataFrame: Separated four levels of productid along with their sessions.
        """
        product_series = (
            data
            .productid
            .str
            .split(';', expand=True)
            .stack()
            .reset_index(level=1, drop=True)
            .str
            .split('/')
        )
        product_df = pd.DataFrame(product_series.to_list(), index=product_series.index)
        product_df.drop(4, axis=1, inplace=True)
        product_df.columns = [f'level{x}' for x in product_df.columns]
        product_df = (
            data
            .reset_index()
            .loc[:, ['index', 'sessionid']]
            .merge(product_df.reset_index(), how='left', on='index')
            .drop('index', axis=1)
        )
        return product_df

    def keep_product(data: pd.DataFrame, level: str, threshold: float = 0.5) -> list:
        """Generate list of productid to be kept at every level. A productid will
        be kept if being satisfied one of following two conditions:
        (1) Appear in more than 100 sessions.
            -> Keep the most common productid.
        (2) Appear in between 10 and 100 sessions and male proportion >= threshold
        (e.g. 0.5) in the sessions having the productid.
            -> Ignore uncommon productid, i.e. appear in fewer than 10 sessions.
            -> For productid neither common or uncommon, i.e. appear in between 10
            and 100 sessions, the productid will be kept only when it has male
            significant more than female compare to the population rate (0.22).

        Args:
            data (pd.DataFrame): Data input.
            level (str): One of ['level0', 'level1', 'level2', 'level3'].
            threshold (float, optional): [description]. Threshold apply to
                considering productid.Defaults to 0.5.

        Returns:
            list: productid to be kept.
        """
        product_df = ShoppingFeatureEngineering.transform_product(data)
        session_count = product_df.groupby(level).sessionid.nunique()
        product_keep = session_count.loc[lambda x: x > 100].index.to_list()
        product_consider = session_count.loc[lambda x: (x >= 10) & (x <= 100)].index.to_list()
        product_add = (
            product_df
            .merge(data[['sessionid', 'male']], how='left', on='sessionid')
            .loc[lambda df: df[level].isin(product_consider)]
            .drop_duplicates(['sessionid', level])
            .groupby(level)
            .male
            .mean()
            .loc[lambda x: x >= threshold]
            .index
            .to_list()
        )
        product_keep.extend(product_add)
        return product_keep

    def gen_feat_product(self):
        product_df = ShoppingFeatureEngineering.transform_product(self.raw)
        for level in ['level0', 'level1', 'level2', 'level3']:
            product_keep = ShoppingFeatureEngineering.keep_product(self.raw, level)
            product_df[level].loc[lambda x: ~x.isin(product_keep)] = np.nan

        product_df = product_df.set_index('sessionid')
        feat_product = pd.get_dummies(product_df, prefix='', prefix_sep='')
        feat_product = feat_product.groupby('sessionid').sum()
        self.feature = pd.concat([self.feature, feat_product], axis=1)

    def gen_data(self) -> pd.DataFrame:
        end = (
            self.raw
            .set_index('sessionid')
            .assign(end=lambda df: pd.to_datetime(df['end']).dt.date)
            .end
        )
        data = pd.concat([end, self.label, self.feature], axis=1)
        return data


if __name__ == '__main__':

    shopping_fe = ShoppingFeatureEngineering()
    shopping_fe.load_raw(
        feature_path='../00_raw/trainingData.csv',
        label_path='../00_raw/trainingLabels.csv')
    shopping_fe.encode_label()
    shopping_fe.gen_feat_time()
    shopping_fe.gen_feat_product()
    data = shopping_fe.gen_data()
    data.to_csv('data.csv')
