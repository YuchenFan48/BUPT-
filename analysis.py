import collections
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# plot the graph of peiji interacting with other countries in the barplot.
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


# The same as countries
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


# Publication of each category
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


# Publication of each year
def analyze_year(df):
    post_2023 = len(df[df['Release Date'].str.startswith('2023')])
    post_2022 = len(df[df['Release Date'].str.startswith('2022')])
    dic = {"2022": post_2022, "2023": post_2023}
    df = pd.DataFrame(list(dic.items()), columns=['Year', 'Count'])
    sns.barplot(x='Year', y='Count', data=df)
    plt.show()
