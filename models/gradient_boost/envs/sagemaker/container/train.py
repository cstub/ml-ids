#!/usr/bin/env python

import sys
import os
import json
import traceback
import uuid
import mlflow

prefix = '/opt/ml/'

output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

input_path = prefix + 'input/data'
training_path = os.path.join(input_path, 'training')
validation_path = os.path.join(input_path, 'validation')
testing_path = os.path.join(input_path, 'testing')

mlflow_project_uri = os.path.join(prefix, 'code/models/gradient_boost/project')
mlflow_out_path = os.path.join('/tmp', str(uuid.uuid4()))


def merge(dict1, dict2):
    d = dict(dict1)
    d.update(dict2)
    return d


if __name__ == '__main__':
    print('Starting the training')

    try:
        with open(param_path, 'r') as tc:
            training_params = json.load(tc)

        training_file_path = os.path.join(training_path, 'train.h5')
        validation_file_path = os.path.join(validation_path, 'val.h5')
        testing_file_path = os.path.join(testing_path, 'test.h5')

        mlflow_params = merge(training_params, {
            'train_path': training_file_path,
            'val_path': validation_file_path,
            'test_path': testing_file_path,
            'output_path': mlflow_out_path,
            'artifact_path': model_path
        })

        os.makedirs(mlflow_out_path, exist_ok=True)

        mlflow.run(mlflow_project_uri, parameters=mlflow_params, use_conda=False)
        print('Training complete.')

        sys.exit(0)
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)
