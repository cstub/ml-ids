"""
CLI to split a single dataset into train/val/test sub-datasets.
"""
import os
import sys
import logging
import click
import pandas as pd
import ml_ids.data.metadata as md
from ml_ids.data.dataset import load_dataset
from ml_ids.model_selection import train_val_test_split

logging.basicConfig(
    format='[%(asctime)s|%(module)s.py|%(levelname)s]  %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout
)


@click.command()
@click.option('--dataset-path', type=click.Path(exists=True), required=True,
              help='Path to the input dataset in .csv format. Can be a folder containing multiple files.')
@click.option('--output-path', type=click.Path(exists=True), required=True,
              help='Path to store the output datasets.')
@click.option('--val-size', type=click.FloatRange(0, 1), default=0.1,
              help='Fraction of the data used for the validation set.')
@click.option('--test-size', type=click.FloatRange(0, 1), default=0.1,
              help='Fraction of the data used for the test set.')
@click.option('--nrows', type=int,
              help='Number of rows to load per input file.')
@click.option('--random-seed', type=int,
              help='Random seed.')
def split_dataset(dataset_path, output_path, val_size, test_size, nrows, random_seed):
    """
    Runs the CLI.
    """
    logging.info('Loading dataset from "%s"...', dataset_path)

    dataset = load_dataset(dataset_path=dataset_path, transform_data=False, nrows=nrows)

    train, val, test = train_val_test_split(dataset,
                                            val_size=val_size,
                                            test_size=test_size,
                                            stratify_col=md.COLUMN_LABEL_CAT,
                                            random_state=random_seed)

    train = remove_extra_labels(train)
    val = remove_extra_labels(val)
    test = remove_extra_labels(test)

    save_dataset(train, output_path, 'train')
    save_dataset(val, output_path, 'val')
    save_dataset(test, output_path, 'test')
    logging.info('Processing complete.')


def remove_extra_labels(dataset: pd.DataFrame):
    """
    Removes unused target labels.
    :param dataset: Input dataset as Pandas DataFrame.
    :return: Dataset without unused target labels.
    """
    return dataset.drop(columns=[md.COLUMN_LABEL_CAT, md.COLUMN_LABEL_IS_ATTACK])


def save_dataset(dataset: pd.DataFrame, path: str, ds_type: str):
    """
    Stores the given dataset in hdf format on the specified path.

    :param dataset: Dataset as Pandas DataFrame.
    :param path: Target path to store the dataset.
    :param ds_type: Dataset type.
    :return: None
    """
    file_path = os.path.join(path, '{}.h5'.format(ds_type))

    logging.info('Storing dataset "%s" of size %d to "%s"', ds_type, len(dataset), file_path)

    dataset.to_hdf(file_path, 'ids_data', format='t', complevel=5, complib='zlib')


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    split_dataset()
