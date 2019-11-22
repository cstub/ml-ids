import json
import click
import logging
from sagemaker.estimator import Estimator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_performance_metric_regex(id):
    """
    Creates the regex for a single performance metric.
    Format: metric_name: 0.12345
    :param id: Metric identifier.
    :return: Regex
    """
    return rf'{id}:\s*([\d.]*)'


def create_metric_def(name, regex):
    """
    Creates a metric definition for a single metric.
    :param name: Metric name.
    :param regex: Metric regex.
    :return: Metric definition as a `dict`.
    """
    return {'Name': name, 'Regex': regex}


def get_metric_definitions():
    """
    Creates the definitions for all metrics to monitor.
    :return: Metric definitions as a `list`.
    """
    return [create_metric_def('train:loss', create_performance_metric_regex('learn')),
            create_metric_def('val:loss', create_performance_metric_regex('test')),
            create_metric_def('val:loss:best', r'bestTest\s=\s([\d.]*)'),
            create_metric_def('test:pr_auc', create_performance_metric_regex('pr_auc')),
            create_metric_def('test:precision', create_performance_metric_regex('precision')),
            create_metric_def('test:recall', create_performance_metric_regex('recall')),
            create_metric_def('test:f1', create_performance_metric_regex('f1'))]


@click.command()
@click.option('--config-path', type=click.Path(exists=True), required=True,
              help='Path to the config.')
@click.option('--param-path', type=click.Path(exists=True), required=True,
              help='Path to the training parameters.')
@click.option('--image-name', type=str, required=True,
              help='Name of the training image')
@click.option('--mode', type=click.Choice(['LOCAL', 'AWS'], case_sensitive=False), default='LOCAL',
              help='Training mode.')
@click.option('--job-id', type=str, required=True,
              help='Unique ID of the training job. Model outputs will be stored in a subdirectory with this name.')
def train(config_path, param_path, image_name, mode, job_id):
    with open(config_path, 'r') as f:
        config = json.load(f)

    with open(param_path, 'r') as f:
        params = json.load(f)

    if mode == 'LOCAL':
        train_instance_type = 'local'
        params['task_type'] = 'CPU'
    else:
        train_instance_type = config['train']['instance_type']
        params['task_type'] = config['train']['task_type']

    train_instance_count = config['train']['instance_count']
    role = config['role']
    model_bucket = config['model_bucket']

    logger.info('Start training with parameters '
                '[job-id="{}", image="{}", mode="{}", instance_type="{}", instance_count={}, params={}]'
                .format(job_id, image_name, mode, train_instance_type, train_instance_count, params))

    estimator = Estimator(image_name=image_name,
                          role=role,
                          train_instance_count=train_instance_count,
                          train_instance_type=train_instance_type,
                          hyperparameters=params,
                          output_path=model_bucket,
                          metric_definitions=get_metric_definitions(),
                          train_max_run=(2 * 60 * 60))

    estimator.fit(job_name=job_id,
                  inputs={
                      'training': config['data']['train'],
                      'validation': config['data']['val'],
                      'testing': config['data']['test']
                  })


if __name__ == '__main__':
    train()
