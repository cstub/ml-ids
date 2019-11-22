import click
import logging
import mlflow
import mlflow.pyfunc
import pickle
import os
import shutil
from catboost import Pool
from ml_ids.data.dataset import load_dataset_hdf
from ml_ids.data.metadata import FEATURES_NO_VARIANCE, FEATURES_TO_IGNORE, FEATURES_PRESERVE_NEG_COLUMNS
from ml_ids.prediction import predict_proba_positive
from ml_ids.model_selection import split_x_y
from ml_ids.models.gradient_boost.train import train_model, GradientBoostHyperParams
from ml_ids.models.gradient_boost.mlflow_wrapper import CatBoostWrapper
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dataset(path):
    """
    Loads a single dataset in `hdf` format.
    :param path: Dataset path.
    :return: Pandas DataFrame.
    """
    return load_dataset_hdf(dataset_path=path,
                            omit_cols=FEATURES_NO_VARIANCE + FEATURES_TO_IGNORE,
                            preserve_neg_value_cols=FEATURES_PRESERVE_NEG_COLUMNS)


def load_train_val_test_dataset(train_path, val_path, test_path):
    """
    Loads the train, validation and test datasets.
    :param train_path: Path to the train dataset.
    :param val_path: Path to the validation dataset.
    :param test_path: Path to the test dataset.
    :return: the `Tuple(train, val, test)` containing Pandas DataFrames.
    """
    return load_dataset(train_path), load_dataset(val_path), load_dataset(test_path)


def measure_performance(clf, pipeline, dataset):
    """
    Measures performance metrics on the given dataset.
    :param clf: Classifier to test.
    :param pipeline: Preprocessing pipeline.
    :param dataset: Dataset.
    :return: the `Tuple(pr_auc, precision, recall, f1)`.
    """
    X, y = split_x_y(dataset)
    X = pipeline.transform(X)

    pool = Pool(X)
    y_true = y.label_is_attack

    pred_proba = predict_proba_positive(clf, pool)
    pred = clf.predict(pool)

    pr_auc = average_precision_score(y_true, pred_proba)
    precision = precision_score(y_true, pred)
    recall = recall_score(y_true, pred)
    f1 = f1_score(y_true, pred)
    return pr_auc, precision, recall, f1


def save_artifacts(cbm_model_path, classifier, pipeline_path, pipeline, col_config_path, column_config):
    """
    Save training artifacts to disk.
    :param cbm_model_path: Path on disk where the classifier should be stored.
    :param classifier: Classifier to store.
    :param pipeline_path: Path on disk where the pipeline should be stored.
    :param pipeline: Pipeline to store.
    :param col_config_path: Path on disk where the config should be stored.
    :param column_config: Column config to store.
    :return: None
    """
    classifier.save_model(cbm_model_path)
    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipeline, f)
    with open(col_config_path, 'wb') as f:
        pickle.dump(column_config, f)


@click.command()
@click.option('--train-path', type=click.Path(exists=True), required=True,
              help='Path to the train dataset in .h5 format.')
@click.option('--val-path', type=click.Path(exists=True), required=True,
              help='Path to the train dataset in .h5 format.')
@click.option('--test-path', type=click.Path(exists=True), required=True,
              help='Path to the train dataset in .h5 format.')
@click.option('--output-path', type=click.Path(exists=True), required=True,
              help='Path to store the output.')
@click.option('--artifact-path', type=click.Path(exists=True), required=True,
              help='Path to store the artifacts.')
@click.option('--use-val-set', type=bool, default=True,
              help='Determines if the evaluation dataset should be used for early stopping of the training process.'
                   'If set to False the evaluation dataset will be appended to the train dataset.')
@click.option('--random-seed', type=int, default=None,
              help='Random seed.')
@click.option('--nr-iterations', type=int, required=True)
@click.option('--tree-depth', type=int, required=True)
@click.option('--l2-reg', type=float, required=True)
@click.option('--border-count', type=int, required=True)
@click.option('--random-strength', type=int, required=True)
@click.option('--task-type', type=click.Choice(['CPU', 'GPU'], case_sensitive=False), required=True)
@click.option('--nr-samples-attack-category', type=int, required=True)
def train(train_path,
          val_path,
          test_path,
          output_path,
          artifact_path,
          use_val_set,
          random_seed,
          nr_iterations,
          tree_depth,
          l2_reg,
          border_count,
          random_strength,
          task_type,
          nr_samples_attack_category):
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)

    cbm_model_path = os.path.join(output_path, 'gradient_boost_model.cbm')
    pipeline_path = os.path.join(output_path, 'preprocessing_pipeline.pkl')
    col_config_path = os.path.join(output_path, 'column_config.pkl')
    mlflow_model_path = os.path.join(artifact_path, 'ml-ids-gb_mlflow_pyfunc')

    random_seed = None if random_seed == -1 else random_seed

    logger.info('Loading datasets...')
    train_dataset, val_dataset, test_dataset = load_train_val_test_dataset(train_path, val_path, test_path)

    if not use_val_set:
        logger.info('Evaluation dataset will not be used for early stopping. Merging with training dataset.')
        train_dataset = train_dataset.append(val_dataset)
        val_dataset = None
    else:
        logger.info('Evaluation dataset will be used for early stopping.')

    hyper_params = GradientBoostHyperParams(nr_iterations=nr_iterations,
                                            tree_depth=tree_depth,
                                            l2_reg=l2_reg,
                                            border_count=border_count,
                                            random_strength=random_strength,
                                            task_type=task_type)

    with mlflow.start_run():
        logger.info('Starting training...')
        clf, pipeline, column_names = train_model(train_dataset,
                                                  val_dataset,
                                                  hyper_params=hyper_params,
                                                  nr_attack_samples=nr_samples_attack_category,
                                                  random_seed=random_seed)

        pr_auc, precision, recall, f1 = measure_performance(clf, pipeline, test_dataset)
        logger.info('Estimator performance:')
        logger.info('pr_auc: %f', pr_auc)
        logger.info('precision: %f', precision)
        logger.info('recall: %f', recall)
        logger.info('f1: %f', f1)

        save_artifacts(cbm_model_path,
                       clf,
                       pipeline_path,
                       pipeline,
                       col_config_path,
                       {
                           'col_names': column_names,
                           'preserve_neg_vals': FEATURES_PRESERVE_NEG_COLUMNS
                       })

        mlflow.pyfunc.save_model(
            path=mlflow_model_path,
            python_model=CatBoostWrapper(),
            artifacts={
                'cbm_model': cbm_model_path,
                'pipeline': pipeline_path,
                'col_config': col_config_path
            },
            conda_env='conda.yaml',
            code_path=['../../../ml_ids'])

        logger.info('Training completed.')


if __name__ == '__main__':
    train()
