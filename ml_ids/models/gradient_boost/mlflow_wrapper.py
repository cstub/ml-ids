"""
Wrapper to enable usage of a CatBoost estimator with MLflow.
"""
import pickle
import mlflow.pyfunc
from catboost import CatBoostClassifier
from ml_ids.data.dataset import remove_negative_values, remove_inf_values


class CatBoostWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow wrapper for CatBoost estimators.
    """

    def load_context(self, context):
        # pylint: disable=attribute-defined-outside-init
        with open(context.artifacts['pipeline'], 'rb') as f:
            self.pipeline = pickle.load(f)

        with open(context.artifacts['col_config'], 'rb') as f:
            column_config = pickle.load(f)

        self.clf = CatBoostClassifier()
        self.clf.load_model(context.artifacts['cbm_model'])
        self.col_names = column_config['col_names']
        self.preserve_cols = column_config['preserve_neg_vals']

    def preprocess(self, data):
        """
        Applies the pre-processing pipeline to the features given in the input dataset.

        :param data: Input dataset.
        :return: Transformed dataset.
        """
        data = data[self.col_names]
        data = remove_inf_values(data)
        data = remove_negative_values(data, ignore_cols=self.preserve_cols)
        return self.pipeline.transform(data)

    def predict(self, context, model_input):
        X = self.preprocess(model_input)
        return self.clf.predict(X)
