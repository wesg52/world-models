import os
import numpy as np
import pandas as pd
import datasets
import torch


def get_decoded_vocab(tokenizer):
    return {
        t: tokenizer.decode(t)
        for t in tokenizer.get_vocab().values()
    }


def make_prompt_dataset(short_prompt, prompt, tokenizer, entity_df, entity_col):
    entity_list = list(entity_df[entity_col].values)

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


def load_entity_data(entity_type):
    data_path = os.path.join(
        'data', 'entity_datasets', f'{entity_type}.csv')
    df = pd.read_csv(data_path)
    return df


def prompt_data_path(entity_type, prompt_name, model_family):
    return os.path.join(
        'data', 'prompt_datasets', model_family, entity_type, prompt_name
    )


def load_tokenized_dataset(entity_type, prompt_name, model_family):
    save_path = prompt_data_path(entity_type, prompt_name, model_family)
    return datasets.load_from_disk(save_path)


class EntityDataManager:
    def __init__(self, entity_type, prompt_dict):
        self.entity_type = entity_type
        self.prompt_dict = prompt_dict
        self.entity_data = None  # DataFrame loaded when needed

    def get_feature_values(self, feature_name, rank_method='dense', return_ranking=False):
        if self.entity_data is None:
            self.entity_data = self.load_entity_data()

        rank_df = self.entity_data[[self.entity_type, feature_name]]

        rank_df = rank_df.dropna()
        rank_df['rank'] = rank_df[feature_name].rank(method=rank_method)
        rank_df = rank_df.set_index(self.entity_type)
        return_column = 'rank' if return_ranking else feature_name
        rank_dict = rank_df[return_column].to_dict()
        return rank_dict

    def make_and_save_tokenized_datasets(self, tokenizer, model_family):
        if self.entity_data is None:
            self.entity_data = self.load_entity_data()

        entity_list = self.entity_data[self.entity_type].tolist()

        if self.entity_type == 'zip':
            entity_list = [str(e).zfill(5) for e in entity_list]

        for short_prompt, full_prompt in self.prompt_dict.items():
            dataset = make_prompt_dataset(
                full_prompt, tokenizer, entity_list)

            save_path = self.prompt_data_path(short_prompt, model_family)
            dataset.save_to_disk(save_path)

    def load_entity_data(self):
        data_path = os.path.join(
            'data', 'entity_datasets', f'{self.entity_type}.csv')
        df = pd.read_csv(data_path)
        return df

    def prompt_data_path(self, prompt_name, model_family):
        return os.path.join(
            'data', 'prompt_datasets', model_family, self.entity_type, prompt_name
        )

    def load_tokenized_dataset(self, prompt_name, model_family):
        save_path = self.prompt_data_path(prompt_name, model_family)
        return datasets.load_from_disk(save_path)
