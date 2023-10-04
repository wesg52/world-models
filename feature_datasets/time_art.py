from .common import *
import random
import pandas as pd

ART_PROMPTS = {
    'empty': '',
    'random': '',
    'release': 'When was the release date of ',
    'empty_all_caps': '',
}



def make_book_entity_df(raw_data_dir, min_wiki_page_views=5000, min_year=1900):
    book_df = pd.read_csv(os.path.join(raw_data_dir, 'books.csv'))
    book_page_view_df = pd.read_csv(
        os.path.join(raw_data_dir, 'book_page_views.csv'),
        names=['entity', 'page_views'], skiprows=1
    )

    book_df['page_name'] = book_df['wikiPage'].apply(
        lambda x: x[len('http://en.wikipedia.org/wiki/'):])
    book_df = book_df.drop(columns=['book', 'wikiPage'])
    book_df = book_df.rename(
        columns={'author': 'creator', 'pageCount': 'length', 'releaseDate': 'release_date'})
    book_df = book_df[book_df.release_date.apply(
        lambda x: x[:4]).astype(int) > min_year]
    book_df = book_df.join(book_page_view_df.set_index('entity'), on='page_name')\
        .sort_values('release_date')\
        .drop_duplicates(['title', 'creator'])\
        .sort_values('page_views', ascending=False)\
        .dropna(subset=['title', 'page_views'])

    book_df = book_df.query('page_views > @min_wiki_page_views')

    book_df = book_df[~book_df.creator.isna()]

    return book_df


def make_movie_entity_df(raw_data_dir, min_wiki_page_views=5000, min_year=1900):
    movie_df = pd.read_csv(os.path.join(raw_data_dir, 'movies.csv'))
    movie_page_view_df = pd.read_csv(
        os.path.join(raw_data_dir, 'movie_page_view.csv'),
        names=['entity', 'page_views'], skiprows=1
    )
    movie_df['page_name'] = movie_df['wikiPage'].apply(
        lambda x: x[len('http://en.wikipedia.org/wiki/'):])
    movie_df = movie_df.drop(columns=['movie', 'wikiPage'])
    movie_df = movie_df.rename(
        columns={'director': 'creator', 'runtime': 'length', 'releaseDate': 'release_date'})
    movie_df = movie_df[movie_df.release_date.apply(
        lambda x: x[:4]).astype(int) > min_year]
    movie_df = movie_df.join(movie_page_view_df.set_index('entity'), on='page_name')\
        .sort_values('release_date')\
        .drop_duplicates(['title', 'creator'])\
        .sort_values('page_views', ascending=False)\
        .dropna(subset=['title', 'page_views'])

    movie_df = movie_df.query('page_views > @min_wiki_page_views')

    movie_df = movie_df[~movie_df.creator.isna()]

    return movie_df


def make_song_entity_df(raw_data_dir, min_wiki_page_views=5000, min_year=1900):
    song_df = pd.read_csv(os.path.join(raw_data_dir, 'songs.csv'))
    song_page_view_df = pd.read_csv(
        os.path.join(raw_data_dir, 'song_page_view.csv'),
        names=['entity', 'page_views'], skiprows=1
    )
    song_df['page_name'] = song_df['wikiPage'].apply(
        lambda x: x[len('http://en.wikipedia.org/wiki/'):])
    song_df = song_df.drop(columns=['song', 'wikiPage'])
    song_df = song_df.rename(
        columns={'artist': 'creator', 'releaseDate': 'release_date'})
    song_df = song_df[song_df.release_date.apply(
        lambda x: x[:4]).astype(int) > min_year]
    song_df = song_df.join(song_page_view_df.set_index('entity'), on='page_name')\
        .sort_values('release_date')\
        .drop_duplicates(['title', 'creator'])\
        .sort_values('page_views', ascending=False)\
        .dropna(subset=['title', 'page_views'])

    song_df = song_df.query('page_views > @min_wiki_page_views')

    song_df = song_df[~song_df.creator.isna()]

    return song_df


def sanitize_title(title):
    try:
        title = title.strip()
        if title[0] == '"':
            title = title[1:]
        if title[-1] == '"':
            title = title[:-1]
        title = title.strip()
        while title[-1] == '.':
            title = title[:-1]
        title = title.strip()
        return title
    except IndexError:
        return ""


def make_art_entity_dataset(raw_data_dir, min_wiki_page_views=5000, min_year=1949, test_ratio=0.2):
    book_df = make_book_entity_df(
        raw_data_dir, min_year=min_year, min_wiki_page_views=min_wiki_page_views)
    movie_df = make_movie_entity_df(
        raw_data_dir, min_year=min_year, min_wiki_page_views=min_wiki_page_views)
    song_df = make_song_entity_df(
        raw_data_dir, min_year=min_year, min_wiki_page_views=min_wiki_page_views)

    book_df['entity_type'] = 'book'
    movie_df['entity_type'] = 'movie'
    song_df['entity_type'] = 'song'

    art_df = pd.concat([book_df, movie_df, song_df])

    art_df['title'] = art_df['title'].apply(sanitize_title)
    art_df = art_df.loc[art_df['title'].apply(lambda x: len(x)) > 1]
    art_df = art_df.reset_index(drop=True)

    unique_creators = art_df.creator.unique()
    n = len(unique_creators)
    test_creators = np.random.choice(
        unique_creators, size=int(n*test_ratio), replace=False)
    test_set = np.array([
        page_name in test_creators for page_name in art_df.creator.values
    ])

    art_df['is_test'] = test_set

    save_path = os.path.join('data', 'entity_datasets', 'art.csv')
    art_df.to_csv(save_path, index=False)


def make_art_prompt_dataset(short_prompt, prompt, tokenizer, art_df):

    dataset_strings = []

    for _, row in art_df.iterrows():
        apos = "'s" if row.creator[-1] != 's' else "'"
        prompt_suffix = f"{row.creator}{apos} {row.title}"

        if short_prompt.endswith('all_caps'):
            prompt_suffix = prompt_suffix.upper()

        dataset_strings.append(prompt + prompt_suffix)

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
        'entity': art_df.title.values.tolist(),
        'creator': art_df.creator.values.tolist(),
        'input_ids': token_ids.tolist(),
        'entity_mask': entity_mask.tolist(),
    })

    dataset.set_format(type='torch', columns=['input_ids'])

    return dataset


class TemporalDataManager(EntityDataManager):
    def __init__(self, entity_type, prompt_dict):
        self.entity_type = entity_type
        self.prompt_dict = prompt_dict
        self.entity_data = None  # DataFrame loaded when needed

    def get_feature_values(self, feature_name):
        if self.entity_data is None:
            self.entity_data = self.load_entity_data()

        time = pd.to_datetime(
            self.entity_data[feature_name],
            format='%Y-%m-%d'
        )
        return time.values.astype(int)

    def make_and_save_tokenized_datasets(self, tokenizer, model_family):
        if self.entity_data is None:
            self.entity_data = self.load_entity_data()

        for short_prompt, full_prompt in self.prompt_dict.items():
            dataset = make_art_prompt_dataset(
                full_prompt, tokenizer, self.entity_data)

            save_path = self.prompt_data_path(short_prompt, model_family)
            dataset.save_to_disk(save_path)
