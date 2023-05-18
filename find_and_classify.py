import openai
import os
import time
import csv
import pkuseg
import collections
import json
import pandas as pd
import re

openai.api_key = 'Enter your own API references'


# use text-davinci-003 to find all the countries appearing in the context
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
    with open('/data/fyc/test/countries.csv', "w", newline='') as f:
        writer = csv.writer(f)
        for country in countries:
            writer.writerow(country)
    return countries


# filter the main country
def post_process_coutries(response):
    if response is None:
        return
    countries = []
    ans = response.choices[0].text.strip()
    ans = ans.split()
    for country in ans:
        if country != '斐济':
            countries.append(country)
    return countries


# The answer given is not standard. Change the form into a more formal one.
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


# find all the international or national countries in the context
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


# Change into a formal form
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


# classify the passsage besed on its context
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


# The category is split into two single words. Combine them into a whole word.
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


# find hot-words in the context
# it has not been finished, we should focus only on words matters.
# Perhaps use BERT tokens or TF/IDF. Too complex.
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


# Change into a formal form
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
