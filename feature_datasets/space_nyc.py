
import numpy as np
import torch
import datasets
from sklearn.model_selection import train_test_split
try:
    import geopandas as gpd
except ImportError:
    pass

# see https://data.cityofnewyork.us/City-Government/Points-Of-Interest/rxuy-2muj
# and https://data.cityofnewyork.us/City-Government/Points-Of-Interest/rxuy-2muj/about

NYC_PLACE_PROMPTS = {
    'empty': '',
    'random': '',
    'where_is': 'Where is ',
    'where_nyc': 'Where in New York City is ',
}


BOROUGH_MAPPING = {
    '1': 'Manhattan',
    '2': 'Bronx',
    '3': 'Brooklyn',
    '4': 'Queens',
    '5': 'Staten Island',
    '6': 'Nassau County',
    '7': 'Westchester',
    '8': 'New Jersey'
}

FACILITY_T_MAPPING = {
    '1': 'Residential',
    '2': 'Education Facility',
    '3': 'Cultural Facility',
    '4': 'Recreational Facility',
    '5': 'Social Services',
    '6': 'Transportation Facility',
    '7': 'Commercial',
    '8': 'Government Facility (non public safety)',
    '9': 'Religious Institution',
    '10': 'Health Services',
    '11': 'Public Safety',
    '12': 'Water',
    '13': 'Miscellaneous'
}


def make_nyc_entity_df(test_size=0.2):
    gdf = gpd.read_file('data/raw_data/nyc_poi.geojson')
    gdf['latitude'] = gdf.geometry.apply(lambda x: x.coords[0][1])
    gdf['longitude'] = gdf.geometry.apply(lambda x: x.coords[0][0])

    # also add nad83 coords (in kms)
    gdf = gdf.to_crs(epsg=2263)
    gdf['nad83_x'] = gdf.geometry.apply(lambda x: x.coords[0][0]) / 3280.84
    gdf['nad83_y'] = gdf.geometry.apply(lambda x: x.coords[0][1]) / 3280.84

    # filters
    gdf = gdf.loc[~gdf.name.str.contains('BUOY')]
    gdf = gdf.loc[~gdf.name.str.contains('BOUY')]

    gdf = gdf.loc[gdf.name.str.len() > 7]

    gdf['borough_name'] = gdf.borough.map(BOROUGH_MAPPING)
    gdf['facility_t_name'] = gdf.facility_t.map(FACILITY_T_MAPPING)

    df = gdf.drop(columns=['geometry']).copy()

    # split train/test by complexid
    unique_cids = gdf.query('complexid != "0"').complexid.unique()
    train_complexids, test_complexids = train_test_split(
        unique_cids, test_size=test_size, random_state=42)
    train_complexids = set(train_complexids)
    test_complexids = set(test_complexids)

    is_test = gdf.complexid.isin(test_complexids)
    frac_test = len(is_test) * test_size
    zero_complex = gdf.query('complexid == "0"').index
    additional_pts = int(frac_test) - sum(is_test)
    zero_complex_test_points = np.random.choice(
        zero_complex, size=additional_pts, replace=False)
    is_test[zero_complex_test_points] = True

    df['is_test'] = is_test

    df.to_csv('data/entity_datasets/nyc_place.csv', index=False)


def normalize_location_names(locations):
    stop_words = {
        'AND', 'OR', 'OF', 'THE', 'A', 'AT', '&', 'IN', 'TO'
    }
    abbv_words = {
        'FDNY', 'NYCT', 'YMCA', 'LGA', 'US', 'NYC', 'PS', 'IS', 'NYS',
        'UN', 'NY', 'EMS', 'JCC', 'NYU', 'CC', 'NYPD', 'NYPA', 'DHS'
    }
    normalized_names = []

    for location in locations:
        words = location.split()
        normalized_name = []
        for i, word in enumerate(words):
            if word.strip() in stop_words:
                normalized_name.append(word.lower())
            elif word.strip() in abbv_words:
                normalized_name.append(word)
            else:
                normalized_name.append(word.lower().capitalize())
        normalized_names.append(' '.join(normalized_name))

    return normalized_names


def make_nyc_prompt_dataset(short_prompt, prompt, tokenizer, entity_df, entity_col):
    entity_list = list(entity_df[entity_col].values)

    entity_list = normalize_location_names(entity_list)

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
