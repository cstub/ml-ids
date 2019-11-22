import click
import json
from mlflow import sagemaker


@click.command()
@click.option('--config-path', type=click.Path(exists=True), required=True,
              help='Path to the config.')
def undeploy(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    app_name = config['deploy']['app_name']
    region = config['deploy']['region']

    sagemaker.delete(app_name=app_name, region_name=region)


if __name__ == '__main__':
    undeploy()
