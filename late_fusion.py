import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import wandb

from config import TASKS, PREDICTION_FOLDER, NUM_TARGETS, MIMIC, PERSONALISATION, AROUSAL, PERSONALISATION_DIMS, HUMOR, WANDB_ENTITY
from main import get_eval_fn


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=TASKS)
    parser.add_argument('--emo_dim', default=AROUSAL, choices=PERSONALISATION_DIMS,
                        help=f'Specify the emotion dimension, only relevant for personalisation (default: {AROUSAL}).')
    parser.add_argument('--model_ids', nargs='+', required=True, help='model ids')
    parser.add_argument('--personalised', nargs='+', required=False,
                        help=f'Personalised model IDs for {PERSONALISATION}, '
                             f'otherwise ignored. Must be the same number as --model_ids')
    parser.add_argument('--seeds', nargs='+', required=False, help=f'seeds, needed for {MIMIC} and {HUMOR}')
    parser.add_argument('--weights', nargs='+', required=False, help='Weights for models', type=float)
    parser.add_argument('--predict', action='store_true')

    args = parser.parse_args()
    assert len(set(args.model_ids)) == len(args.model_ids), "Error, duplicate model file"
    assert len(args.model_ids) >= 2, "For late fusion, please give at least 2 different models"

    if args.weights and args.task != 'mimic':
        assert len(args.weights) == len(args.model_ids)
    elif args.weights and args.task == 'mimic':
        assert len(args.weights) == len(args.model_ids) or len(args.weights) == NUM_TARGETS['mimic'] * len(
            args.model_ids)

    if args.task == PERSONALISATION:
        assert len(args.model_ids) == len(args.personalised)
        assert args.emo_dim
    else:
        assert args.seeds

    if args.seeds and len(args.seeds) == 1:
        args.seeds = [args.seeds[0]] * len(args.model_ids)
        assert len(args.model_ids) == len(args.seeds)
    if args.task == PERSONALISATION:
        # os.path.join(config.PREDICTION_FOLDER, PERSONALISATION, 'personalised',args.emo_dim,args.model_id, args.run_name)
        args.prediction_dirs = [
            os.path.join(PREDICTION_FOLDER, PERSONALISATION, 'personalised', args.emo_dim, args.model_ids[i],
                         args.personalised[i]) for
            i in range(len(args.model_ids))]
    else:
        args.prediction_dirs = [os.path.join(PREDICTION_FOLDER, args.task, args.model_ids[i], args.seeds[i]) for i in
                                range(len(args.model_ids))]
    return args


def create_pers_lf(df, weights=None):
    pred_arr = df[[c for c in df.columns if c.startswith('pred')]].values
    if weights is None:
        weights = [1.] * pred_arr.shape[1]
    weights = np.array(weights) / np.sum(weights)
    for i in range(weights.shape[0]):
        preds = pred_arr[:, i]
        preds = weights[i] * preds
        pred_arr[:, i] = preds
    fused_preds = np.sum(pred_arr, axis=1)
    labels = df['label_gs'].values
    if args.predict:
        fused_df = pd.DataFrame({'fused_predictions': fused_preds, 'labels': labels})
        fused_df.to_csv(f"fused_predictions_{wandb.run.name}.csv", index=False)
    print(fused_preds)
    return fused_preds, labels


import re


def extract_contents(text):
    # This regex pattern matches contents within square brackets
    pattern = r'\[([^\]]+)\]'

    # Find all matching patterns
    matches = re.findall(pattern, text)

    # If there are at least two contents, return the first and second
    if len(matches) >= 2:
        return matches[0], matches[1]
    else:
        return None, None


if __name__ == '__main__':
    args = parse_args()
    wandb.init(entity=WANDB_ENTITY, project='Late_fusion')
    model_ids = args.model_ids
    features = []
    sample = model_ids[0]
    emodim = extract_contents(sample)[1]
    model_types = ['RNN' if model_id.startswith('RNN') else 'TF' for model_id in model_ids]

    for model_id in model_ids:
        features.append(extract_contents(model_id)[0])
    feature_with_model = []
    for idx, feature in enumerate(features):
        feature_with_model.append(f"{feature}({model_types[idx]})")
    wandb.run.name = f"{emodim}_{'+'.join(feature_with_model)}"
    wandb.log({'emodim': emodim, 'features': features})
    for partition in ['devel', 'test']:
        dfs = [pd.read_csv(os.path.join(pred_dir, f'predictions_{partition}.csv')) for pred_dir in args.prediction_dirs]

        meta_cols = [c for c in list(dfs[0].columns) if c.startswith('meta_')]
        for meta_col in meta_cols:
            assert all(np.all(df[meta_col].values == dfs[0][meta_col].values) for df in dfs)
        meta_df = dfs[0][meta_cols].copy()

        label_cols = [c for c in list(dfs[0].columns) if c.startswith('label')]
        # assert all(np.all(df[label_col].values == dfs[0][label_col].values) for df in dfs)
        # ignore due to test set missing
        label_df = dfs[0][label_cols].copy()

        prediction_dfs = []
        for i, df in enumerate(dfs):
            pred_df = df.drop(columns=meta_cols + label_cols)
            pred_df.rename(columns={c: f'{c}_{args.model_ids[i]}' for c in pred_df.columns}, inplace=True)
            prediction_dfs.append(pred_df)
        prediction_df = pd.concat(prediction_dfs, axis='columns')

        full_df = pd.concat([meta_df, prediction_df, label_df], axis='columns')
        if args.predict:
            full_df.to_csv(f"for_check_{wandb.run.name}.csv", index=False)
            def late2pred(df):
                # Extract the columns that need to be averaged
                pred_columns = [col for col in df.columns if col not in ['meta_subj_id', 'label_gs']]

                # Calculate the average of the pred_columns
                df['pred'] = df[pred_columns].mean(axis=1)

                # Select only the meta_subj_id, pred and label_gs columns
                new_df = df[['meta_subj_id', 'pred', 'label_gs']]
                new_df.to_csv(f"predict_{wandb.run.name}.csv", index=False)

                return new_df
            # late2pred(full_df)

        if args.task == 'personalisation':
            preds, labels = create_pers_lf(full_df, weights=args.weights)

        if args.task != MIMIC and partition == 'devel':
            eval_fn, eval_str = get_eval_fn(args.task)

            result = np.round(eval_fn(preds, labels), 4)
            print(f'{partition}: {result} {eval_str}')
            wandb.log({'result': result})
