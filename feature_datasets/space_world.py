from .common import *
import random
import pandas as pd
import re

PLACE_PROMPTS = {
    'empty': '',
    'empty_all_caps': '',
    'random': '',
    'coords': 'What are the lat/lon coordinates of ',
}


def remove_lower_specificity_duplicates(entity_df, column_name):
    specificity_ordering = entity_df.type.value_counts().sort_values()
    specificity_ordering = specificity_ordering.index.tolist()

    # sort by specificity
    entity_df['order'] = pd.Categorical(
        entity_df['type'], categories=specificity_ordering, ordered=True)
    entity_df = entity_df.sort_values('order')

    # drop duplicates, keeping the first (most specific) entry
    entity_df = entity_df.drop_duplicates(subset=[column_name], keep='first')

    return entity_df.drop(columns=['order'])


def make_structure_entity_df(raw_data_dir, min_wiki_page_views=5000):
    structure_page_views = pd.read_csv(
        os.path.join(raw_data_dir, 'structure_page_view.csv'))
    structure_df = pd.read_csv(os.path.join(raw_data_dir, 'structures.csv'))
    structure_df = remove_lower_specificity_duplicates(
        structure_df, 'landmark_name')

    structure_df['page_name'] = structure_df['wikipedia_link'].apply(
        lambda x: x[len('http://en.wikipedia.org/wiki/'):])

    structure_df = structure_df.join(structure_page_views.fillna(
        0).set_index('wiki_title'), on='page_name')
    structure_df['country'] = structure_df['country'].apply(
        lambda x: x.split('/')[-1])
    structure_df['type'] = structure_df['type'].apply(
        lambda x: x.split('/')[-1])
    structure_df = structure_df.rename(columns={'landmark_name': 'name'})
    structure_df = structure_df.drop(
        columns=['landmark', 'wikipedia_link', 'page_name'])

    structure_df = structure_df.query('page_views > @min_wiki_page_views')

    return structure_df


def make_natural_place_entity_df(raw_data_dir, min_wiki_page_views=5000):
    natural_place_page_views = pd.read_csv(
        os.path.join(raw_data_dir, 'natural_place_page_view.csv'))
    natural_place_df = pd.read_csv(
        os.path.join(raw_data_dir, 'natural_places.csv'))
    natural_place_df = remove_lower_specificity_duplicates(
        natural_place_df, 'place_name')

    natural_place_df['page_name'] = natural_place_df['wikipedia_link'].apply(
        lambda x: x[len('http://en.wikipedia.org/wiki/'):])

    natural_place_df = natural_place_df.join(
        natural_place_page_views.fillna(0).set_index('wiki_title'), on='page_name')
    natural_place_df['country'] = natural_place_df['country'].apply(
        lambda x: x.split('/')[-1])
    natural_place_df['type'] = natural_place_df['type'].apply(
        lambda x: x.split('/')[-1])
    natural_place_df = natural_place_df.rename(columns={'place_name': 'name'})
    natural_place_df = natural_place_df.drop(
        columns=['place', 'wikipedia_link', 'page_name'])

    natural_place_df = natural_place_df.query(
        'page_views > @min_wiki_page_views')

    return natural_place_df


def make_populated_place_entity_df(raw_data_dir, min_wiki_page_views=5000):
    populated_place_page_views = pd.read_csv(
        os.path.join(raw_data_dir, 'populated_place_page_view.csv'))
    populated_place_df = pd.read_csv(
        os.path.join(raw_data_dir, 'populated_places.csv'))
    populated_place_df = remove_lower_specificity_duplicates(
        populated_place_df, 'place_name')

    populated_place_df['page_name'] = populated_place_df['wikipedia_link'].apply(
        lambda x: x[len('http://en.wikipedia.org/wiki/'):])

    populated_place_df = populated_place_df.join(
        populated_place_page_views.fillna(0).set_index('wiki_title'), on='page_name')
    populated_place_df['country'] = populated_place_df['country'].apply(
        lambda x: x.split('/')[-1])
    populated_place_df['type'] = populated_place_df['type'].apply(
        lambda x: x.split('/')[-1])
    populated_place_df = populated_place_df.rename(
        columns={'place_name': 'name'})
    populated_place_df = populated_place_df.drop(
        columns=['place', 'wikipedia_link', 'page_name'])

    populated_place_df = populated_place_df.query(
        'population < 10**9')  # filter erroneous population values

    populated_place_df = populated_place_df.query(
        'page_views > @min_wiki_page_views')

    return populated_place_df


def make_world_landmark_entity_dataset(raw_data_dir, min_wiki_page_views=5000, test_ratio=0.2):
    structure_df = make_structure_entity_df(
        raw_data_dir, min_wiki_page_views=min_wiki_page_views)
    populated_place_df = make_populated_place_entity_df(
        raw_data_dir, min_wiki_page_views=min_wiki_page_views)
    natural_place_df = make_natural_place_entity_df(
        raw_data_dir, min_wiki_page_views=min_wiki_page_views)

    structure_df['entity_type'] = 'structure'
    populated_place_df['entity_type'] = 'populated_place'
    natural_place_df['entity_type'] = 'natural_place'

    landmark_df = pd.concat(
        [structure_df, populated_place_df, natural_place_df])

    landmark_df = landmark_df.reset_index(drop=True)

    landmark_df = landmark_df.rename(columns={'type': 'entity_subtype'})

    # choose test indices (needs to be updated if repeated entities are allowed)
    n = len(landmark_df)
    test_ixs = np.random.choice(n, size=int(n*test_ratio), replace=False)
    test_set = np.zeros(n, dtype=bool)
    test_set[test_ixs] = True

    landmark_df['is_test'] = test_set

    save_path = os.path.join('data', 'entity_datasets', 'world_place.csv')
    landmark_df.to_csv(save_path, index=False)


def move_text_within_parentheses(input_str):
    # Find text within parentheses using regular expression
    match = re.search(r'\((.*?)\)', input_str)

    # Check if there's any text within parentheses
    if match:
        text_within_parentheses = match.group(1)

        # Remove text within parentheses from the original string
        input_str = re.sub(r'\((.*?)\)', '', input_str)

        # Prepend the text followed by a 's' and the rest of the string
        output_str = f"{text_within_parentheses}'s {input_str}"

        return output_str, True

    return input_str, False


def make_world_prompt_dataset(short_prompt, prompt, tokenizer, entity_df, entity_col):
    entity_list = list(entity_df[entity_col].values)

    normalized_names = []
    for name in entity_list:
        name, processed = move_text_within_parentheses(name)
        if not processed and ',' in name:
            splits = name.split(',')
            name = f"{splits[-1].strip()}'s {','.join(splits[:-1])}"
        normalized_names.append(name)

    if short_prompt.endswith('all_caps'):
        normalized_names = [name.upper() for name in normalized_names]

    dataset_strings = [prompt + entity for entity in normalized_names]

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
        'entity': normalized_names,
        'input_ids': token_ids.tolist(),
        'entity_mask': entity_mask.tolist(),
    })

    dataset.set_format(type='torch', columns=['input_ids'])

    return dataset


class SpatialDataManager(EntityDataManager):
    def __init__(self, entity_type, prompt_dict):
        self.entity_type = entity_type
        self.prompt_dict = prompt_dict
        self.entity_data = None  # DataFrame loaded when needed

    def get_feature_values(self, feature_name):
        if self.entity_data is None:
            self.entity_data = self.load_entity_data()

        return self.entity_data[feature_name].values

    def make_and_save_tokenized_datasets(self, tokenizer, model_family):
        if self.entity_data is None:
            self.entity_data = self.load_entity_data()

        entity_list = self.entity_data['name'].tolist()

        for short_prompt, full_prompt in self.prompt_dict.items():
            dataset = make_prompt_dataset(
                full_prompt, tokenizer, entity_list)

            save_path = self.prompt_data_path(short_prompt, model_family)
            dataset.save_to_disk(save_path)


COUNTRY_CONTINENTS = {  # thanks Claude!
    'Greece': 'Europe',
    'United_States': 'North America',
    'Egypt': 'Africa',
    'Monaco': 'Europe',
    'Maldives': 'Asia',
    'Canada': 'North America',
    'Chile': 'South America',
    'Norway': 'Europe',
    'Italy': 'Europe',
    'Spain': 'Europe',
    'France': 'Europe',
    'Comoros': 'Africa',
    'Australia': 'Oceania',
    'New_Zealand': 'Oceania',
    'Senegal': 'Africa',
    'Honduras': 'North America',
    'Mexico': 'North America',
    'Solomon_Islands': 'Oceania',
    'Iceland': 'Europe',
    'Sweden': 'Europe',
    'Thailand': 'Asia',
    'Indonesia': 'Asia',
    'Bahrain': 'Asia',
    'Scotland': 'Europe',
    'Japan': 'Asia',
    'Bahamas': 'North America',
    'Denmark': 'Europe',
    'Russia': 'Asia',
    'Tonga': 'Oceania',
    'Vanuatu': 'Oceania',
    'Croatia': 'Europe',
    'Republic_of_Ireland': 'Europe',
    'Fiji': 'Oceania',
    'Seychelles': 'Africa',
    'Estonia': 'Europe',
    'Russian_Federation': 'Asia',
    'Kiribati': 'Oceania',
    'Republic_of_Kiribati': 'Oceania',
    'Papua_New_Guinea': 'Oceania',
    'Tuvalu': 'Oceania',
    'India': 'Asia',
    'Philippines': 'Asia',
    'Cook_Islands': 'Oceania',
    'Solomon_Islands_(archipelago)': 'Oceania',
    'Portugal': 'Europe',
    'Saudi_Arabia': 'Asia',
    'China': 'Asia',
    'Bailiwick_of_Guernsey': 'Europe',
    'Azerbaijan': 'Asia',
    'Myanmar': 'Asia',
    'Marshall_Islands': 'Oceania',
    'Mauritius': 'Africa',
    'Dominican_Republic': 'North America',
    'South_Korea': 'Asia',
    'Kingdom_of_Denmark': 'Europe',
    'Kent': 'Europe',
    'Cuba': 'North America',
    'Brazil': 'South America',
    'Puerto_Rico': 'North America',
    'Peru': 'South America',
    'British_Antarctic_Territory': 'Antarctica',
    'Trinidad_and_Tobago': 'North America',
    'Hong_Kong': 'Asia',
    'French_Republic': 'Europe',
    'United_Kingdom': 'Europe',
    'Vietnam': 'Asia',
    'Grenada': 'North America',
    'Wales': 'Europe',
    'Falkland_Islands': 'South America',
    'Switzerland': 'Europe',
    'U.S._Virgin_Islands': 'North America',
    "People's_Republic_of_China": 'Asia',
    'Guyana': 'South America',
    'Dominica': 'North America',
    'Ecuador': 'South America',
    'Netherlands': 'Europe',
    'Malta_(island)': 'Europe',
    'Kuwait': 'Asia',
    'Cape_Verde': 'Africa',
    'Germany': 'Europe',
    'Bangladesh': 'Asia',
    'Greenland': 'North America',
    'South_Africa': 'Africa',
    'Sri_Lanka': 'Asia',
    'Federated_States_of_Micronesia': 'Oceania',
    'United_Arab_Emirates': 'Asia',
    'Eritrea': 'Africa',
    'Finland': 'Europe',
    'Namibia': 'Africa',
    'Kingdom_of_the_Netherlands': 'Europe',
    'Isle_of_Man': 'Europe',
    'Cambodia': 'Asia',
    'Iran': 'Asia',
    'Malaysia': 'Asia',
    'Republic_of_China': 'Asia',
    'Guinea-Bissau': 'Africa',
    'Nicaragua': 'North America',
    'England': 'Europe',
    'Northern_Ireland': 'Europe',
    'British_Indian_Ocean_Territory': 'Asia',
    'Panama': 'North America',
    'São_Tomé_and_Príncipe': 'Africa',
    'Montenegro': 'Europe',
    'Argentina': 'South America',
    'Tunisia': 'Africa',
    'Ireland': 'Europe',
    'Ghana': 'Africa',
    'Bulgaria': 'Europe',
    'Democratic_Republic_of_the_Congo': 'Africa',
    'Samoa': 'Oceania',
    'Venezuela': 'South America',
    'Yemen': 'Asia',
    'Jamaica': 'North America',
    'Barbados': 'North America',
    'Bermuda': 'North America',
    'Cyprus': 'Asia',
    'Nauru': 'Oceania',
    'Saint_Lucia': 'North America',
    'Saint_Vincent_and_the_Grenadines': 'North America',
    'Qatar': 'Asia',
    'Hawaii': 'Oceania',
    'Kenya': 'Africa',
    'Saint_Kitts_and_Nevis': 'North America',
    'South_Georgia_and_the_South_Sandwich_Islands': 'Antarctica',
    'British_Virgin_Islands': 'North America',
    'Oman': 'Asia',
    'The_Bahamas': 'North America',
    'Palau': 'Oceania',
    'Antarctica': 'Antarctica',
    'Malta': 'Europe',
    'Poland': 'Europe',
    'Singapore': 'Asia',
    'Jersey': 'Europe',
    'Latvia': 'Europe',
    'Haiti': 'North America',
    'Antigua': 'North America',
    'Taiwan': 'Asia',
    'Antigua_&_Barbuda': 'North America',
    'United_States_of_America': 'North America',
    'Tanzania': 'Africa',
    'Liberia': 'Africa',
    'Belize': 'North America',
    'Colombia': 'South America',
    'East_Timor': 'Asia',
    'Albania': 'Europe',
    'American_Samoa': 'Oceania',
    'Faroe_Islands': 'Europe',
    'Malawi': 'Africa',
    'Kazakhstan': 'Asia',
    'Costa_Rica': 'North America',
    'Bosnia_and_Herzegovina': 'Europe',
    'Equatorial_Guinea': 'Africa',
    'Brunei': 'Asia',
    'Ethiopia': 'Africa',
    'Uganda': 'Africa',
    'Sierra_Leone': 'Africa',
    'Togo': 'Africa',
    'Zambia': 'Africa',
    'Afghanistan': 'Asia',
    'Guinea': 'Africa',
    'Hungary': 'Europe',
    'Turkey': 'Asia',
    'Pakistan': 'Asia',
    'Burundi': 'Africa',
    'Burkina_Faso': 'Africa',
    'Lebanon': 'Asia',
    'Rwanda': 'Africa',
    'Nepal': 'Asia',
    'Czech_Republic': 'Europe',
    'Lithuania': 'Europe',
    'Morocco': 'Africa',
    'El_Salvador': 'North America',
    'North_Korea': 'Asia',
    'Uzbekistan': 'Asia',
    'Guatemala': 'North America',
    'Mongolia': 'Asia',
    'Zimbabwe': 'Africa',
    'Armenia': 'Asia',
    'Austria': 'Europe',
    'Mauritania': 'Africa',
    'Romania': 'Europe',
    'Djibouti': 'Africa',
    'Bolivia': 'South America',
    'Ambazonia': 'Africa',
    'Turkmenistan': 'Asia',
    'Nevis': 'North America',
    'Madagascar': 'Africa',
    'Kosovo': 'Europe',
    'Libya': 'Africa',
    'Guernsey': 'Europe',
    'Sahrawi_Arab_Democratic_Republic': 'Africa',
    'Tajikistan': 'Asia',
    'Uruguay': 'South America',
    'Ivory_Coast': 'Africa',
    'Andorra': 'Europe',
    'Somaliland': 'Africa',
    'Georgia_(country)': 'Asia',
    'Bhutan': 'Asia',
    'Chad': 'Africa',
    'Benin': 'Africa',
    'Otuke': 'Africa',
    'Kyrgyzstan': 'Asia',
    'Somalia': 'Africa',
    'Suriname': 'South America',
    'Lesotho': 'Africa',
    'Mozambique': 'Africa',
    'Moldova': 'Europe',
    'Gabon': 'Africa',
    'Israel': 'Asia',
    'Serbia': 'Europe',
    'Saint_Kitts': 'North America',
    'Ukraine': 'Europe',
    'Liechtenstein': 'Europe',
    'Kingdom_of_Romania': 'Europe',
    'Antigua_and_Barbuda': 'North America',
    'Channel_Islands': 'Europe',
    'Republic_of_China_(Taiwan)': 'Asia',
    'Republic_of_the_Congo': 'Africa',
    'British_Raj': 'Asia',
    'North_Macedonia': 'Europe',
    'Belgium': 'Europe',
    'Cameroon': 'Africa',
    'Slovakia': 'Europe',
    'Angola': 'Africa',
    'Iraq': 'Asia',
    'South_Sudan': 'Africa',
    'Belarus': 'Europe',
    'Syria': 'Asia',
    'Canada_(New_France)': 'North America',
    'North_Rustico,_Prince_Edward_Island': 'North America',
    'German_Democratic_Republic': 'Europe',
    'Slovenia': 'Europe',
    'Algeria': 'Africa',
    'Sudan': 'Africa',
    'Kingdom_of_Italy': 'Europe',
    'Ethiopian_Empire': 'Africa',
    'Paraguay': 'South America',
    'Caribbean': 'North America',
    'Botswana': 'Africa',
    'Gloucestershire': 'Europe',
    'DR_Congo': 'Africa',
    'Leicestershire': 'Europe',
    'Nigeria': 'Africa',
    'Mali': 'Africa',
    'West_Sussex': 'Europe',
    'Oxfordshire': 'Europe',
    'Jordan': 'Asia',
    'State_of_Palestine': 'Asia',
    'Staffordshire': 'Europe',
    'Hampshire': 'Europe',
    'Surrey': 'Europe',
    'Cambridgeshire': 'Europe',
    'North_Yorkshire': 'Europe',
    'Bosnia_&_Herzegovina': 'Europe',
    'Timor-Leste': 'Asia',
    'Mandatory_Palestine': 'Asia',
    'Lancashire': 'Europe',
    'Luxembourg': 'Europe',
    'Somerset': 'Europe',
    'Palestinian_Territories': 'Asia',
    'Niger': 'Africa',
    'Central_African_Republic': 'Africa',
    'Nottinghamshire': 'Europe',
    'Warwickshire': 'Europe',
    'Cumbria': 'Europe',
    'Norfolk': 'Europe',
    'Crow_Nation': 'North America',
    'Essex': 'Europe',
    'Worcestershire': 'Europe',
    'Suffolk': 'Europe',
    'Devon': 'Europe',
    'Myanmar_(Burma)': 'Asia',
    'Eswatini': 'Africa',
    'Derbyshire': 'Europe',
    'Lincolnshire': 'Europe',
    'Hertfordshire': 'Europe',
    'Laos': 'Asia',
    'Ngau_Tau_Kok': 'Asia',
    'Somalia_': 'Africa',
    'Rutland': 'Europe',
    'East_Sussex': 'Europe',
    'Old_Cairo': 'Africa',
    'Judith_Basin_County,_Montana': 'North America',
    'England_national_rugby_union_team': 'Europe',
    'San_Marino': 'Europe',
    'Niue': 'Oceania',
    'Mogilev_Region': 'Europe',
    'Buganda': 'Africa',
    'Russian_Empire': 'Asia',
    'Northamptonshire': 'Europe',
    'Gambia': 'Africa',
    'The_Gambia': 'Africa',
    'Dorset': 'Europe',
    'Danish_Realm': 'Europe',
    'United_Mexican_States': 'North America',
    'Western_Sahara': 'Africa',
    'Hokkaido': 'Asia',
    'Minsk_Raion': 'Europe',
    'ملف:Flag_of_Sudan.svg': 'Africa',
    'Republic_of_landmarksakh': 'Asia',
    'De_facto': '',
    'landmarksakh': 'Asia',
    'England_Rugby': 'Europe',
    'Terra_nullius': '',
    'Puntland': 'Africa',
    'Aruba': 'North America',
    'Kwun_Tong': 'Asia',
    'Dutch_East_Indies': 'Asia',
    'Burma': 'Asia',
    'Kainai_Nation': 'North America',
    'Aysén_del_General_Carlos_Ibáñez_del_Campo_Region': 'South America',
    'Rosebud_Sioux_Tribe': 'North America',
    'County_Durham': 'Europe',
    'india': 'Asia',
    'Los_Ríos_Region': 'South America',
    'Hudson_County': 'North America',
    'Hopi': 'North America',
    'Republic_of_India': 'Asia',
    'De_facto': 'Europe',
    'Terra_nullius': 'Africa',
}
