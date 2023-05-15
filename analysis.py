from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import numpy as np
import torch
import os
import openai
import time
import json
import csv
import pkuseg
import collections
import re
import seaborn as sns
import matplotlib.pyplot as plt


os.environ['TRANSFORMERS_CACHE'] = '/data/fyc'

openai.api_key = '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'


def post_process_coutries(response):
    if response is None:
        return
    countries = []
    ans = response.choices[0].text.strip()
    ans = ans.split()
    for country in ans:
        if country != '斐济':
            countries.append(country)
    print(countries)
    return countries


def find_countries(df,
                   output_dir='/data/fyc/test',
                   model_name='text-davinci-003',
                   top_p=1.0):
    dict = []
    dict = df[['Title', 'Context']].to_dict('records')

    prompt = '给出下面文章中出现的所有国家的名称。'
    inputs = []
    for item in dict:
        inputs.append('题目:' + item['Title'] + '\n' +
                      '文章:' + item['Context'] + '\n' + prompt)
    for input in inputs:
        input.replace('?', "")
    os.makedirs(output_dir, exist_ok=True)
    countries = []
    for input in inputs:
        begin_time = time.time()
        response = openai.Completion.create(
            engine=model_name,
            prompt=input,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0,
            top_p=top_p,
            frequency_penalty=0
        )
        result = post_process_coutries(response)
        countries.append(result)
        end_time = time.time()
        print(f'Request time lasts for {end_time - begin_time}')
    with open('/data/fyc/test/countries.json', "w", encoding='utf-8') as f:
        json.dump(countries, f)
    with open('/data/fyc/test/countries.csv', "w", newline='') as f:
        writer = csv.writer(f)
        for country in countries:
            writer.writerow(country)
    return countries


def find_organization(df,
                      output_dir='/data/fyc/test',
                      model_name='text-davinci-003',
                      top_p=1.0):
    dict = []
    dict = df[['Title', 'Context']].to_dict('records')

    prompt = '根据下面给出的文章（包括题目和文章），给出出现的全球和部分地区组织或计划，如联合国、亚太计划等。只需要给出组织名。国家并不被认为是一个组织,如美国、韩国并不是一个组织。只需要给出组织或计划，不需要其他信息。'
    inputs = []
    for item in dict:
        inputs.append('题目:' + item['Title'] + '\n' +
                      '文章:' + item['Context'] + '\n' + prompt)
    for input in inputs:
        input.replace('?', "")
    os.makedirs(output_dir, exist_ok=True)
    organizations = []
    for input in inputs:
        begin_time = time.time()
        response = openai.Completion.create(
            engine=model_name,
            prompt=input,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0,
            top_p=top_p,
            frequency_penalty=0
        )
        result = post_process_coutries(response)
        organizations.append(result)
        end_time = time.time()
        print(f'Request time lasts for {end_time - begin_time}')
        with open('/data/fyc/test/organization.csv', "w", newline='') as f:
            writer = csv.writer(f)
            for organization in organizations:
                writer.writerow(organization)
    return organizations


def find_category(df,
                  output_dir='/data/fyc/test',
                  model_name='text-davinci-003',
                  top_p=1.0,
                  temperature=0):
    dict = []
    dict = df[['Title', 'Context']].to_dict('records')
    prompt = "根据给定的题目和文本，给出此报告属于文化、教育、历史、旅游中的哪一类，你只需要回答是哪一类就可以。答案应该属于文化、教育历史、旅游中的一个。如果不属于任何一类，你可以回答正确的分类。所有的答案只包含分类，不需要其他信息。"
    inputs = []
    for item in dict:
        inputs.append('题目:' + item['Title'] + '\n' +
                      '文章:' + item['Context'] + '\n' + prompt)
    for input in inputs:
        input.replace('?', "")
    os.makedirs(output_dir, exist_ok=True)
    categories = []
    for input in inputs:
        begin_time = time.time()
        response = openai.Completion.create(
            engine=model_name,
            prompt=input,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=0
        )
        result = response.choices[0].text.strip()
        categories.append(result)
        end_time = time.time()
        print(result)
        print(f'The request lasts for {end_time - begin_time}s')
    with open('/data/fyc/test/category.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for category in categories:
            writer.writerow(category)


def find_hot_words(df):
    dict = []
    dict = df[['Title', 'Context']].to_dict('records')
    inputs = []
    for item in dict:
        input = "题目: " + item['Title'] + "内容: " + item['Context']
        inputs.append(input)
    for input in inputs:
        seg = pkuseg.pkuseg()
        words = seg.cut(input)
        filtered_words = [word for word in words if len(word) > 1]
        black_list = ['斐济', '我们', '他们', '她们', '它们']
        filtered_words = [
            word for word in filtered_words if word not in black_list]
        words_count = collections.Counter(filtered_words)
        hot_words = words_count.most_common(5)
        with open('/data/fyc/test/hot_words.json', 'a', encoding='utf-8') as f:
            json.dump(hot_words, f, ensure_ascii=False)
        with open('/data/fyc/test/hot_words.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(hot_words)

# finish


def post_process_category():
    with open('/data/fyc/test/category.csv', 'r') as f:
        reader = csv.reader(f)
        cleaned_categories = []
        for row in reader:
            category = row
            cleaned_category = ''
            for word in category:
                cleaned_category += word
            if '：' in cleaned_category:
                cleaned_category = cleaned_category.split('：')
                cleaned_category = cleaned_category[1]
            cleaned_categories.append(cleaned_category)
    df = pd.DataFrame(cleaned_categories, columns=['Category'])
    df.to_csv('/data/fyc/test/cleaned_category.csv')


# finish


def process_countries():
    with open('/data/fyc/test/countries.csv', 'r') as f:
        reader = csv.reader(f)
        cleaned_countries = []
        for row in reader:
            countries = ""
            for country in row:
                countries += country
            if "：" in countries:
                country_list = countries.split('：')
                if '10个太平洋国家' not in country_list:
                    countries = "".join(country_list[1:])
                else:
                    countries = "".join(country_list[2:])
            countries = re.sub(r'\d+\.\s*', ' ', countries)
            countries = countries.strip()
            countries = countries.replace('、', ' ')
            countries = countries.replace('斐济', '')
            cleaned_countries.append(countries)
        df = pd.DataFrame(cleaned_countries, columns=["Countries mentioned"])
        df.to_csv('/data/fyc/test/cleaned_country.csv')


def post_process_organizations():
    with open('/data/fyc/test/organization.csv', 'r') as f:
        reader = csv.reader(f)
        cleaned_organizations = []
        for row in reader:
            organizations = ""
            for organization in row:
                organizations += organization
            if "：" in organizations:
                organization_list = organizations.split('：')
                organizations = "".join(organization_list)
            organizations = re.sub(r'\d+\.\s*', ' ', organizations)
            organizations = organizations.strip()
            organizations = "".join(set(organizations.split()))
            organizations = organizations.replace('、', ' ')
            cleaned_organizations.append(organizations)
    df = pd.DataFrame(cleaned_organizations, columns=["Organizations"])
    df.to_csv('/data/fyc/test/cleaned_organization.csv')


def hotwords_df():
    with open('/data/fyc/test/hot_words.csv', 'r') as f:
        reader = csv.reader(f)
        hot_words = []
        for row in reader:
            words = ""
            for tuple in row:
                pos = tuple.find(',')
                word = tuple[2: pos - 1]
                words = words + word + " "
            hot_words.append(words)
        print(hot_words)
        df = pd.DataFrame(hot_words, columns=['Top5-Hot-Words'])
        df.to_csv('/data/fyc/test/cleaned_hot_words.csv')


def load_data():
    path_cleaned_category = '/data/fyc/test/cleaned_category.csv'
    path_cleaned_country = '/data/fyc/test/cleaned_country.csv'
    path_cleaned_hot_words = '/data/fyc/test/cleaned_hot_words.csv'
    path_cleaned_organization = '/data/fyc/test/cleaned_organization.csv'
    path_cleaned_date = '/data/fyc/test/cleaned_date.csv'
    path_url = '/data/fyc/test/url.csv'
    df_category = pd.read_csv(path_cleaned_category, encoding='utf-8')
    df_country = pd.read_csv(path_cleaned_country, encoding='utf-8')
    df_hot_words = pd.read_csv(path_cleaned_hot_words, encoding='utf-8')
    df_organization = pd.read_csv(path_cleaned_organization, encoding='utf-8')
    df_date = pd.read_csv(path_cleaned_date, encoding='utf-8')
    urls = []
    with open(path_url, 'r', encoding='gb2312', errors='replace') as f:
        idx = 0
        for line in f.readlines():
            idx += 1
            if idx == 2 or idx == 1:
                continue
            if idx == 530:
                break
            pos = line.find('h')
            url = line[pos:]
            urls.append(url)
    df_url = pd.DataFrame(urls, columns=['url'])
    df_url.to_csv('/data/fyc/test/cleaned_url.csv')
    return df_category, df_country, df_hot_words, df_organization, df_date, df_url


def analyze_countries(df):
    countries = df['Countries mentioned'].str.split(
        ',').explode().unique().tolist()
    all_countries = []
    for country in countries:
        if country != 'Not mentioned':
            country_list = country.split()
        for item in country_list:
            all_countries.append(item)
    hash = collections.Counter(all_countries)
    country_in_total = len(hash)
    popular_country = [{k: v} for k, v in hash.items() if v >= 5]
    df = pd.DataFrame(popular_country)
    df = df.melt(var_name='Country', value_name='Count').dropna()
    print(df)
    sns.catplot(x='Country', y='Count', data=df, kind='bar')
    plt.show()


def process_data():
    df = pd.read_csv('/data/fyc/test/data.csv',
                     encoding='utf-8', encoding_errors='ignore')
    df['Length'] = df['Context'].fillna('').apply(len)
    Q1 = np.percentile(df['Length'], 10)
    Q3 = np.percentile(df['Length'], 95)
    df = df[df['Length'] >= int(Q1)]  # 记得>306
    df = df[df['Length'] <= int(Q3)]
    df = df.reset_index(drop=True)
    df_category, df_country, df_hot_words, df_organization, df_date, df_url = load_data()
    df["Category"] = df_category["Category"]
    df["Countries mentioned"] = df_country["Countries mentioned"].fillna(
        "Not mentioned")
    df["Top5-Hot-Words"] = df_hot_words["Top5-Hot-Words"]
    df["Organizations"] = df_organization["Organizations"]
    df["URL"] = df_url["url"]
    df = df[df['Length'] > 306].reset_index(drop=True)
    df["Release Date"] = df_date["Date"]
    df.to_csv('/data/fyc/test/cleaned.csv')
    return df


def analyze_countries(df):
    countries = df['Countries mentioned'].str.split(
        ',').explode().unique().tolist()
    df_2023 = df[df['Release Date'].str.startswith('2023')]
    countries_2023 = df_2023['Countries mentioned'].str.split(
        ',').explode().unique().tolist()
    df_2022 = df[df['Release Date'].str.startswith('2022')]
    countries_2022 = df_2022['Countries mentioned'].str.split(
        ',').explode().unique().tolist()
    all_countries = []
    all_countries_2023 = []
    all_countries_2022 = []
    for country in countries:
        if country != 'Not mentioned':
            country_list = country.split()
        for item in country_list:
            all_countries.append(item)
    for country in countries_2023:
        if country != 'Not mentioned':
            country_list = country.split()
        for item in country_list:
            all_countries_2023.append(item)
    for country in countries_2022:
        if country != 'Not mentioned':
            country_list = country.split()
        for item in country_list:
            all_countries_2022.append(item)
    hash = collections.Counter(all_countries)
    hash_2023 = collections.Counter(all_countries_2023)
    hash_2022 = collections.Counter(all_countries_2022)
    country_in_total = len(hash)
    popular_country = [{k: v} for k, v in hash.items() if v >= 20]
    popular_country_2023 = [{k: v} for k, v in hash_2023.items() if v >= 20]
    popular_country_2022 = [{k: v} for k, v in hash_2022.items() if v >= 20]
    df = pd.DataFrame(popular_country)
    df_2023 = pd.DataFrame(popular_country_2023)
    df_2022 = pd.DataFrame(popular_country_2022)
    df = df.melt(var_name='Country', value_name='Count').dropna()
    df_2023 = df_2023.melt(var_name='Country', value_name='Count').dropna()
    df_2022 = df_2022.melt(var_name='Country', value_name='Count').dropna()
    plt.rcParams['font.sans-serif'] = ['Songti SC']
    plt.rcParams['axes.unicode_minus'] = False
    ax1 = sns.barplot(x='Country', y='Count', data=df)
    ax1.set_xticklabels(ax1.get_xticklabels(), size=4)
    plt.title('Both 2023 and 2022')
    plt.show()
    ax2 = sns.barplot(x='Country', y='Count', data=df_2023)
    ax2.set_xticklabels(ax2.get_xticklabels(), size=4)
    plt.title('2023 only')
    plt.show()
    ax3 = sns.barplot(x='Country', y='Count', data=df_2022)
    ax3.set_xticklabels(ax3.get_xticklabels(), size=4)
    plt.title('2022 only')
    plt.show()


def analyze_organizations(df):
    organizations = df['Organizations'].str.split(
        ',').explode().unique().tolist()
    df_2023 = df[df['Release Date'].str.startswith('2023')]
    organizations_2023 = df_2023['Organizations'].str.split(
        ',').explode().unique().tolist()
    df_2022 = df[df['Release Date'].str.startswith('2022')]
    organizations_2022 = df_2022['Organizations'].str.split(
        ',').explode().unique().tolist()
    all_organizations = []
    all_organizations_2023 = []
    all_organizations_2022 = []
    for organization in organizations:
        if organization != 'Not mentioned':
            organization_list = organization.split()
        for item in organization_list:
            all_organizations.append(item)
    for organization in organizations_2023:
        if organizations != 'Not mentioned':
            organization_list = organization.split()
        for item in organization_list:
            all_organizations_2023.append(item)
    for organization in organizations_2022:
        if organization != 'Not mentioned':
            organization_list = organization.split()
        for item in organization_list:
            all_organizations_2022.append(item)
    hash = collections.Counter(all_organizations)
    hash_2023 = collections.Counter(all_organizations_2023)
    hash_2022 = collections.Counter(all_organizations_2022)
    organization_in_total = len(hash)
    popular_organization = [{k: v} for k, v in hash.items() if v >= 15]
    popular_organization_2023 = [{k: v}
                                 for k, v in hash_2023.items() if v >= 10]
    popular_organization_2022 = [{k: v}
                                 for k, v in hash_2022.items() if v >= 8]
    print(popular_organization)
    df = pd.DataFrame(popular_organization)
    df_2023 = pd.DataFrame(popular_organization_2023)
    df_2022 = pd.DataFrame(popular_organization_2022)
    df = df.melt(var_name='Organizations', value_name='Count').dropna()
    df_2023 = df_2023.melt(var_name='Organizations',
                           value_name='Count').dropna()
    df_2022 = df_2022.melt(var_name='Organizations',
                           value_name='Count').dropna()
    plt.rcParams['font.sans-serif'] = ['Songti SC']
    plt.rcParams['axes.unicode_minus'] = False
    ax1 = sns.barplot(x='Organizations', y='Count', data=df)
    ax1.set_xticklabels(ax1.get_xticklabels(), size=4)
    plt.title('Both 2023 and 2022')
    plt.show()
    ax2 = sns.barplot(x='Organizations', y='Count', data=df_2023)
    ax2.set_xticklabels(ax2.get_xticklabels(), size=4)
    plt.title('2023 only')
    plt.show()
    ax3 = sns.barplot(x='Organizations', y='Count', data=df_2022)
    ax3.set_xticklabels(ax3.get_xticklabels(), size=4)
    plt.title('2022 only')
    plt.show()


def analyze_category(df):
    dic = {"历史": 0, "文化": 0,
           "旅游": 0, "教育": 0, "其他": 0}
    dic_2023 = {"历史": 0, "文化": 0,
                "旅游": 0, "教育": 0, "其他": 0}
    dic_2022 = {"历史": 0, "文化": 0,
                "旅游": 0, "教育": 0, "其他": 0}
    category = ['历史', '文化', '旅游', '教育']
    df['Category'] = df['Category'].apply(
        lambda x: x if x in category else '其他')
    df_2023 = df[df['Release Date'].str.startswith('2023')]
    df_2023['Category'] = df['Category'].apply(
        lambda x: x if x in category else '其他')
    df_2022 = df[df['Release Date'].str.startswith('2022')]
    df_2022['Category'] = df['Category'].apply(
        lambda x: x if x in category else '其他')
    category_counts = df['Category'].value_counts()
    category_counts_2023 = df_2023['Category'].value_counts()
    category_counts_2022 = df_2022['Category'].value_counts()
    dic['历史'] += category_counts.get('历史', 0)
    dic['文化'] += category_counts.get('文化', 0)
    dic['旅游'] += category_counts.get('旅游', 0)
    dic['教育'] += category_counts.get('教育', 0)
    dic['其他'] += category_counts.get('其他', 0)
    dic_2023['历史'] += category_counts_2023.get('历史', 0)
    dic_2023['文化'] += category_counts_2023.get('文化', 0)
    dic_2023['旅游'] += category_counts_2023.get('旅游', 0)
    dic_2023['教育'] += category_counts_2023.get('教育', 0)
    dic_2023['其他'] += category_counts_2023.get('其他', 0)
    dic_2022['历史'] += category_counts_2022.get('历史', 0)
    dic_2022['文化'] += category_counts_2022.get('文化', 0)
    dic_2022['旅游'] += category_counts_2022.get('旅游', 0)
    dic_2022['教育'] += category_counts_2022.get('教育', 0)
    dic_2022['其他'] += category_counts_2022.get('其他', 0)
    plt.rcParams['font.sans-serif'] = ['Songti SC']
    plt.rcParams['axes.unicode_minus'] = False
    df = pd.DataFrame(list(dic.items()), columns=[
                      'Category', 'Count'])
    df_2023 = pd.DataFrame(list(dic_2023.items()), columns=[
                           'Category', 'Count'])
    df_2022 = pd.DataFrame(list(dic_2022.items()), columns=[
                           'Category', 'Count'])
    ax1 = sns.lineplot(x='Category', y='Count', data=df,
                       label='Both 2023 and 2022')
    ax2 = sns.lineplot(x='Category', y='Count',
                       data=df_2023, label='2023 only')
    ax3 = sns.lineplot(x='Category', y='Count',
                       data=df_2022, label='2022 only')
    plt.legend(loc='best')
    plt.show()


def analyze_year(df):
    post_2023 = len(df[df['Release Date'].str.startswith('2023')])
    post_2022 = len(df[df['Release Date'].str.startswith('2022')])
    dic = {"2022": post_2022, "2023": post_2023}
    df = pd.DataFrame(list(dic.items()), columns=['Year', 'Count'])
    sns.barplot(x='Year', y='Count', data=df)
    plt.show()


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


df = process_data()
split_category(df)
