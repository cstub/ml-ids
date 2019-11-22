import json
import click
import mlflow
import shutil
import os


def merge(dict1, dict2):
    """
    Merges two dictionaries by creating copies of the dictionaries.
    :param dict1: First dictionary to merge
    :param dict2: Second dictionary to merge
    :return: Merged dictionary
    """
    d = dict(dict1)
    d.update(dict2)
    return d


@click.command()
@click.option('--train-path', type=click.Path(exists=True), required=True,
              help='Path to the train dataset in .h5 format.')
@click.option('--val-path', type=click.Path(exists=True), required=True,
              help='Path to the train dataset in .h5 format.')
@click.option('--test-path', type=click.Path(exists=True), required=True,
              help='Path to the train dataset in .h5 format.')
@click.option('--output-path', type=click.Path(), required=True,
              help='Path to store the output.')
@click.option('--param-path', type=click.Path(exists=True), required=True,
              help='Path to the training parameters.')
def train(train_path, val_path, test_path, output_path, param_path):
    with open(param_path, 'r') as f:
        params = json.load(f)

    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)

    run_params = merge(params, {
        'train_path': train_path,
        'val_path': val_path,
        'test_path': test_path,
        'output_path': output_path,
        'artifact_path': output_path,
    })

    mlflow.run('models/gradient_boost/project',
               parameters=run_params)


if __name__ == '__main__':
    train()
