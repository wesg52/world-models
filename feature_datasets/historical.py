from .common import *
import random
import pandas as pd

HISTORICAL_PROMPTS = {
    'empty': '',
    'random': '',
    'when': 'When did ',
    'when_all_caps': 'When did ',
}


def make_historical_figure_entity_df(top_per_decade=350, year_cutoff=-1000, test_ratio=0.2):
    df = pd.read_csv('data/raw_data/AgeDataset-V1.csv')
    df = df.rename(columns={
        'Id': 'wiki_id', 'Name': 'name', 'Short description': 'description', 'Gender': 'gender',
        'Country': 'country', 'Occupation': 'occupation', 'Birth year': 'birth_year',
        'Death year': 'death_year', 'Manner of death': 'cause_of_death', 'Age of death': 'age'
    })

    df['death_century'] = df['death_year'] // 100
    df['death_decade'] = df['death_year'] // 10
    top_df = df.groupby('death_decade').head(top_per_decade)
    top_df = top_df[top_df['death_year'] > year_cutoff]
    top_df.occupation.fillna('', inplace=True)
    top_df['occupation'] = top_df.occupation.str.lower()

    n = len(top_df)
    test_ixs = np.random.choice(n, size=int(n*test_ratio), replace=False)
    test_set = np.zeros(n, dtype=bool)
    test_set[test_ixs] = True

    top_df['is_test'] = test_set

    top_df.to_csv('data/entity_datasets/historical_figure.csv', index=False)


def make_historical_figure_prompt_dataset(short_prompt, prompt, tokenizer, entity_df):
    entity_df['occupation'] = entity_df['occupation'].fillna('')
    dataset_strings = []
    for _, row in entity_df.iterrows():
        entity_string = ''
        if short_prompt.endswith('occupation'):
            add_space = len(row['occupation']) > 0
            entity_string += row['occupation'] + (' ' if add_space else '')
        entity_string += row['name']

        if short_prompt.endswith('all_caps'):
            entity_string = entity_string.upper()

        dataset_strings.append(prompt + entity_string)

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
        'entity': entity_df.name.values.tolist(),
        'input_ids': token_ids.tolist(),
        'entity_mask': entity_mask.tolist(),
    })

    dataset.set_format(type='torch', columns=['input_ids'])

    return dataset
