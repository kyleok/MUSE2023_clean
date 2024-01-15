# Team FeelsGood: MuSe-Stress 2023 Personalisation, Transformer Encoder

[Homepage](https://www.muse-challenge.org) || [Baseline Paper](https://www.researchgate.net/publication/370100318_The_MuSe_2023_Multimodal_Sentiment_Analysis_Challenge_Mimicked_Emotions_Cross-Cultural_Humour_and_Personalisation)


## Introduction

This git contains the MuSe 2023 participating team FeelsGood output. We leverage a Transformer Encoder model, and organized it to be compatible with the existing code as much as possible. For details about competition, please see the [Baseline Paper](https://www.researchgate.net/publication/370100318_The_MuSe_2023_Multimodal_Sentiment_Analysis_Challenge_Mimicked_Emotions_Cross-Cultural_Humour_and_Personalisation).

If you would like to see our approach and its results please see [Our paper](https://dl.acm.org/doi/pdf/10.1145/3606039.3613104) 


## Installation
It is highly recommended to run everything in a Python virtual environment. Please make sure to install the packages listed 
in ``requirements.txt`` and adjust the paths in `config.py` (especially ``BASE_PATH``). 

You can then e.g. run the unimodal baseline reproduction calls in the ``*.sh`` file provided for each sub-challenge.

## Settings
The ``main.py`` script is used for training and evaluating models. Most important options:
* ``--model_type``: choose either `RNN` or `TF`
* ``--feature``: choose a feature set provided in the data (in the ``PATH_TO_FEATURES`` defined in ``config.py``). Adding 
``--normalize`` ensures normalization of features (recommended for eGeMAPS features).
* Options defining the model architecture: ``model_dim``, ``rnn_n_layers``, ``rnn_bi``, ``d_fc_out``
* Options for the training process: ``--epochs``, ``--lr``, ``--seed``,  ``--n_seeds``, ``--early_stopping_patience``,
``--reduce_lr_patience``,   ``--rnn_dropout``, ``--linear_dropout``
* In order to use a GPU, please add the flag ``--use_gpu``


For more details, please see the ``parse_args()`` method in ``main.py``. 


## Citation:
```bibtex
@inproceedings{10.1145/3606039.3613104,
author = {Park, Ho-Min and Kim, Ganghyun and Van Messem, Arnout and De Neve, Wesley},
title = {MuSe-Personalization 2023: Feature Engineering, Hyperparameter Optimization, and Transformer-Encoder Re-Discovery},
year = {2023},
isbn = {9798400702709},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3606039.3613104},
doi = {10.1145/3606039.3613104},
abstract = {This paper presents our approach for the MuSe-Personalization sub-challenge of the fourth Multimodal Sentiment Analysis Challenge (MuSe 2023), with the goal of detecting human stress levels through multimodal sentiment analysis. We leverage and enhance a Transformer-encoder model, integrating improvements that mitigate issues related to memory leakage and segmentation faults. We propose novel feature extraction techniques, including a pose feature based on joint pair distance and self-supervised learning-based feature extraction for audio using Wav2Vec2.0 and Data2Vec. To optimize effectiveness, we conduct extensive hyperparameter tuning. Furthermore, we employ interpretable meta-learning to understand the importance of each hyperparameter. The outcomes obtained demonstrate that our approach excels in personalization tasks, with particular effectiveness in Valence prediction. Specifically, our approach significantly outperforms the baseline results, achieving an Arousal CCC score of 0.8262 (baseline: 0.7450), a Valence CCC score of 0.8844 (baseline: 0.7827), and a combined CCC score of 0.8553 (baseline: 0.7639) on the test set. These results secured us the second place in MuSe-Personalization.},
booktitle = {Proceedings of the 4th on Multimodal Sentiment Analysis Challenge and Workshop: Mimicked Emotions, Humour and Personalisation},
pages = {89–97},
numpages = {9},
keywords = {multimodal sentiment analysis, emotion detection, multimodal fusion, human pose},
location = {<conf-loc>, <city>Ottawa ON</city>, <country>Canada</country>, </conf-loc>},
series = {MuSe '23}
}
```
