import click
import json
import boto3
import tarfile
import re
import logging
from mlflow import sagemaker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def unpack(file):
    """
    Unpacks compressed files of format `tar` and `tar.gz`.
    :param file: Filename.
    :return: None
    """
    if file.endswith("tar.gz"):
        tar = tarfile.open(file, "r:gz")
        tar.extractall()
        tar.close()
    elif file.endswith("tar"):
        tar = tarfile.open(file, "r:")
        tar.extractall()
        tar.close()


@click.command()
@click.option('--config-path', type=click.Path(exists=True), required=True,
              help='Path to the config.')
@click.option('--job-id', type=str, required=True,
              help='Unique ID of the training job. Model is retrieved from a subdirectory with this name.')
def deploy(config_path, job_id):
    with open(config_path, 'r') as f:
        config = json.load(f)

    app_name = config['deploy']['app_name']
    instance_type = config['deploy']['instance_type']
    instance_count = config['deploy']['instance_count']
    region = config['deploy']['region']
    role = config['role']
    model_name = config['model_name']
    model_bucket = re.sub('s3://', '', config['model_bucket'])
    model_artifact = config['model_artifact']
    model_path = '{}/output/{}'.format(job_id, model_artifact)

    logger.info('Deploying model with parameters '
                '[app-name="{}", instance-type="{}", instance-count={}, region="{}", model-path="{}"]'
                .format(app_name, instance_type, instance_count, region, model_path))

    s3 = boto3.client('s3')
    s3.download_file(model_bucket, model_path, model_artifact)

    unpack(model_artifact)

    sagemaker.deploy(app_name=app_name,
                     model_uri=model_name,
                     execution_role_arn=role,
                     region_name=region,
                     mode='replace',
                     instance_type=instance_type,
                     instance_count=instance_count)


if __name__ == '__main__':
    deploy()
