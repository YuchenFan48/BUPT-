import pandas as pd
import numpy as np

'''
    all the csv files should be the utf-8 encoding.
    Otherwise, try to use with open(... errors = 'replace') to fix.
'''


def process_url():
    with open('/data/fyc/test/url.csv', 'r', encoding='utf-8', errors='replace') as f:
        df = pd.read_csv(f)
    df_url = df[["Title", "URL"]]
    df_url.to_csv('/data/fyc/test/cleaned_url.csv', encoding='utf-8')
    return df


# The url is not matched with the data. So we use the title to combine the two DataFrames
def combine_url(df, df_url):
    merged_df = pd.merge(df, df_url, on='Title', how='left')
    return merged_df


def load_data():
    path_cleaned_category = '/data/fyc/test/cleaned_category.csv'
    path_cleaned_country = '/data/fyc/test/cleaned_country.csv'
    path_cleaned_hot_words = '/data/fyc/test/cleaned_hot_words.csv'
    path_cleaned_organization = '/data/fyc/test/cleaned_organization.csv'
    path_cleaned_date = '/data/fyc/test/cleaned_date.csv'
    path_url = '/data/fyc/test/cleaned_url.csv'
    df_category = pd.read_csv(path_cleaned_category, encoding='utf-8')
    df_country = pd.read_csv(path_cleaned_country, encoding='utf-8')
    df_hot_words = pd.read_csv(path_cleaned_hot_words, encoding='utf-8')
    df_organization = pd.read_csv(path_cleaned_organization, encoding='utf-8')
    df_date = pd.read_csv(path_cleaned_date, encoding='utf-8')
    df_url = pd.read_csv(path_url, encoding='utf-8')
    return df_category, df_country, df_hot_words, df_organization, df_date, df_url


# get the cleaned data.
def process_data():
    df = pd.read_csv('/data/fyc/test/data.csv',
                     encoding='utf-8', encoding_errors='ignore')
    df['Length'] = df['Context'].fillna('').apply(len)
    df_category, df_country, df_hot_words, df_organization, df_date, df_url = load_data()
    Q1 = np.percentile(df['Length'], 10)
    Q3 = np.percentile(df['Length'], 95)
    df = df[df['Length'] >= int(Q1)]
    df = df[df['Length'] <= int(Q3)]
    df = df.reset_index(drop=True)
    df["Category"] = df_category["Category"]
    df["Countries mentioned"] = df_country["Countries mentioned"].fillna(
        "Not mentioned")
    df["Top5-Hot-Words"] = df_hot_words["Top5-Hot-Words"]
    df["Organizations"] = df_organization["Organizations"]
    df = df.drop(['Unnamed: 2'], axis=1)
    df = df[df['Length'] > 306].reset_index(drop=True)
    df["Release Date"] = df_date["Date"]
    df = combine_url(df, df_url)
    print(df)
    df.to_csv('/data/fyc/test/cleaned.csv')
    return df


# Based on the cleaned data and split the data into five sub csv.
def split_category(df):
    category = ['历史', '文化', '旅游', '教育']
    df['Category'] = df['Category'].apply(
        lambda x: x if x in category else '其他')
    grouped_df = df.groupby('Category')
    for name, group in grouped_df:
        if name == '历史':
            group.to_csv('/data/fyc/test/history.csv')
        elif name == '文化':
            group.to_csv('/data/fyc/test/culture.csv')
        elif name == '旅游':
            group.to_csv('/data/fyc/test/travel.csv')
        elif name == '教育':
            group.to_csv('/data/fyc/test/educatiom.csv')
        else:
            group.to_csv('/data/fyc/test/others.csv')
