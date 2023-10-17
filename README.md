# World Models in LLMs
Official code repository for the paper "Language Models Represent Space and Time" by Wes Gurnee and Max Tegmark.

This repository contains all experimental infrastructure for the paper. We expect most users to just be interested in the cleaned data CSVs containing entity names and relevant metadata. These can be found in `data/entity_datasets/` (with the tokenized versions for Llama and Pythia models available in the `data/prompt_datasets/` folder for each prompt type).

In the coming weeks we will release a minimal version of the code to run basic probing experiments on our datasets.



## Cite
If you found our code our datasets helpful in your research, please cite our paper
```
@article{gurnee2023language,
  title={Language Models Represent Space and Time},
  author={Gurnee, Wes and Tegmark, Max},
  journal={arXiv preprint arXiv:2310.02207},
  year={2023}
}
```
