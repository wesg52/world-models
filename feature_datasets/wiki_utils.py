import requests
import os
import json
import time
import tqdm
import pickle
import datetime
import numpy as np
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON


def timestamp():
    return datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")


def query_str_to_df(query_str):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery(query_str)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    rows = results['results']['bindings']

    # Simplify the structure and create a list of flattened dictionaries
    flattened_data = []
    for b in rows:
        flattened_data.append({k: v['value'] for k, v in b.items()})

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(flattened_data)
    return df


def query_entity_all(query_generator_fn, limit=5000):
    offset = 0
    page_dfs = []

    while True:
        query_str = query_generator_fn(offset, limit)
        page_df = query_str_to_df(query_str)
        page_dfs.append(page_df)
        print(f'{timestamp()} offset: {offset}, len(page_df): {len(page_df)}')
        offset += limit

        if len(page_df) < limit:
            break

    return pd.concat(page_dfs)


HEADERS = {
    'User-Agent': 'World models (wesg@mit.edu)'
}


def get_page_views(title, start_date, end_date):
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{title}/monthly/{start_date}/{end_date}"
    response = requests.get(url, headers=HEADERS)
    data = json.loads(response.text)

    total_views = 0
    try:
        for item in data['items']:
            total_views += item['views']
    except KeyError:
        return np.nan

    return total_views


def get_page_view_dict(page_list, start_date, end_date, sleep_dur=0.01, log_every=1000):
    if os.path.exists('temp_page_view_dict.pkl'):
        with open('temp_page_view_dict.pkl', 'rb') as handle:
            cache_dict = pickle.load(handle)
            print(f'Loaded {len(cache_dict)} cached page views')
    else:
        cache_dict = {}

    page_view_dict = {}
    for ix, page in tqdm.tqdm(enumerate(page_list)):
        if page in cache_dict:
            page_view_dict[page] = cache_dict[page]
        else:
            page_view_dict[page] = get_page_views(page, start_date, end_date)
            time.sleep(sleep_dur)

        if ix % log_every == 0:
            with open('temp_page_view_dict.pkl', 'wb') as handle:
                pickle.dump(page_view_dict, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

    return page_view_dict


def make_page_view_dataset(entity_df, entity_name, start_date, end_date, page_column_name='wikiPage'):
    prefix = 'http://en.wikipedia.org/wiki/'
    page_list = [wiki_url[len(prefix):]
                 for wiki_url in entity_df[page_column_name].values]
    unique_page_list = list(set(page_list))
    print(f'n entities: {len(page_list)} | n_unique: {len(unique_page_list)}')

    page_view_dict = get_page_view_dict(unique_page_list, start_date, end_date)

    view_df = pd.Series(page_view_dict).to_frame().reset_index().rename(
        columns={'index': 'wiki_title', 0: 'page_views'})

    SAVE_PATH = os.path.join('data', 'raw_data')
    view_df.to_csv(os.path.join(
        SAVE_PATH, f'{entity_name}_page_view.csv'), index=False)
    return view_df
