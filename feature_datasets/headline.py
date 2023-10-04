import pandas as pd
import numpy as np
from .common import *

# source: https://www.kaggle.com/datasets/johnbandy/new-york-times-headlines
# Good example
# Pilgrims, Fewer and Socially Distanced, Arrive in Mecca for Annual Hajj

HEADLINE_PROMPTS = {
    'empty': '',
    'empty_wo_period': '',
    'when_w_period': 'Publication date of: ',
    'when_wo_period': 'Publication date of: ',
}


def make_headline_prompt_dataset(short_prompt, prompt, tokenizer, entity_df):
    entity_list = list(entity_df['headline'].values)
    if short_prompt.endswith('wo_period'):
        dataset_strings = [prompt + entity[:-1] for entity in entity_list]
    else:
        dataset_strings = [prompt + entity for entity in entity_list]

    token_ids = tokenizer.batch_encode_plus(
        dataset_strings,
        return_tensors='pt',
        padding=True,
        add_special_tokens=False,
        return_attention_mask=False
    )['input_ids']

    if short_prompt == 'random':
        random_prompts = torch.randint(
            low=100, high=token_ids.max().item(),
            size=(token_ids.shape[0], 10),
            dtype=torch.long
        )
        token_ids = torch.cat([random_prompts, token_ids], dim=1)

    # add bos token
    token_ids = torch.cat([
        torch.ones(token_ids.shape[0], 1,
                   dtype=torch.long) * tokenizer.bos_token_id,
        token_ids], dim=1
    )

    prompt_tokens = (token_ids[0] == token_ids).all(axis=0)
    entity_mask = torch.ones_like(token_ids, dtype=torch.bool)
    entity_mask[:, prompt_tokens] = False
    entity_mask[token_ids == tokenizer.pad_token_id] = False

    dataset = datasets.Dataset.from_dict({
        'entity': entity_list,
        'input_ids': token_ids.tolist(),
        'entity_mask': entity_mask.tolist(),
    })

    dataset.set_format(type='torch', columns=['input_ids'])

    return dataset


SECTION_PAIRS = [
    ('Foreign', 'World'),
    ('National', 'U.S.'),
    ('Politics', 'U.S.'),
    ('Washington', 'U.S.'),
    ('Obits', 'all')
]

TOPICS = [
    'trump',
    'biden',
    'coronavirus',
    'covid',
    'obama',
    'clinton',
    'iraq',
    'afghanistan',
    'china',
    'iran'
]

KEEP_COLS = [
    'headline', 'word_count', 'pub_date',
    'print_section', 'print_page', 'section', 'news_desk', 'year'
]


def process_year_df(article_df):
    article_df = article_df.query("type == 'article'")

    section_dfs = []
    for news_desk, section in SECTION_PAIRS:
        if section == 'all':
            df = article_df.loc[article_df.news_desk == news_desk]
        else:
            df = article_df.loc[
                (article_df.news_desk == news_desk) &
                (article_df.section == section)
            ]
        section_dfs.append(df)
    filtered_df = pd.concat(section_dfs)
    filtered_df = filtered_df.loc[filtered_df.isna().sum(axis=1) == 0]
    filtered_df = filtered_df.loc[~filtered_df.headline.str.contains('\\?')]
    try:
        filtered_df = filtered_df.loc[
            filtered_df.print_page.str.isnumeric() == True]
    except AttributeError:
        print('Print page is float')
    filtered_df.print_page = filtered_df.print_page.astype(int)
    filtered_df = filtered_df.loc[
        ((filtered_df.print_page <= 10) & (filtered_df.news_desk != 'Foreign')) |
        ((filtered_df.print_page <= 5) & (filtered_df.news_desk == 'Foreign'))
    ]
    filtered_df = filtered_df.loc[~filtered_df.headline.str.endswith('.')]
    filtered_df = filtered_df[KEEP_COLS]

    # check if titles contain topics
    for topic in TOPICS:
        is_topic = filtered_df.headline.str.lower().str.contains(topic)
        filtered_df[f'is_{topic}'] = is_topic

    return filtered_df


def make_headlines_entity_df(test_ratio=0.2, years=tuple(list(range(2010, 2021)))):
    year_dfs = []
    for year in years:
        print(year)
        article_df = pd.read_csv(
            f'data/raw_data/nyt_articles/new_york_times_stories_{year}.csv')
        year_df = process_year_df(article_df)
        year_dfs.append(year_df)
    full_df = pd.concat(year_dfs)

    full_df['headline'] = full_df.headline.apply(lambda x: x + '.')

    # choose test indices (needs to be updated if repeated entities are allowed)
    n = len(full_df)
    test_ixs = np.random.choice(n, size=int(n*test_ratio), replace=False)
    test_set = np.zeros(n, dtype=bool)
    test_set[test_ixs] = True
    full_df['is_test'] = test_set

    full_df.to_csv('data/entity_datasets/headline.csv', index=False)
