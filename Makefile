SAGEMAKER_TRAIN_CONFIG_PATH=models/gradient_boost/envs/sagemaker/configs/train-gpu.json
SAGEMAKER_DEPLOY_CONFIG_PATH=models/gradient_boost/envs/sagemaker/configs/deploy.json
TRAIN_PARAM_PATH=models/gradient_boost/training_params.json
TRAIN_PATH=dataset/train.h5
VAL_PATH=dataset/val.h5
TEST_PATH=dataset/test.h5

clean:
	-rm -r -f build
	mkdir build

test:
	python -m pytest tests

lint:
	pylint ml_ids

lint-errors:
	pylint ml_ids -E

typecheck:
	mypy ml_ids

split_dataset:
	mkdir -p dataset
	python ./ml_ids/data/split_dataset.py \
		--dataset-path $(DATASET_PATH) \
		--output-path dataset \
		--random-seed 42

train_local:
	python ./models/gradient_boost/envs/local/train.py \
		--train-path $(TRAIN_PATH) \
		--val-path $(VAL_PATH) \
		--test-path $(TEST_PATH) \
		--output-path build/models/gradient_boost \
		--param-path $(TRAIN_PARAM_PATH)

sagemaker_build_image:
	./models/gradient_boost/envs/sagemaker/scripts/build_image.sh ml-ids-train-sagemaker $(TAG)

sagemaker_push_image:
	./models/gradient_boost/envs/sagemaker/scripts/push_image_to_ecr.sh ml-ids-train-sagemaker $(TAG) | grep -Po '(?<=^image-name=).*' > sagemaker-image-name.txt

sagemaker_train_local:
	python ./models/gradient_boost/envs/sagemaker/scripts/train.py \
  		--config-path $(SAGEMAKER_TRAIN_CONFIG_PATH) \
  		--param-path $(TRAIN_PARAM_PATH) \
  		--mode LOCAL \
  		--image-name "ml-ids-train-sagemaker:$(TAG)" \
  		--job-id "ml-ids-sagemaker-job"

sagemaker_train_aws:
	python ./models/gradient_boost/envs/sagemaker/scripts/train.py \
  		--config-path $(SAGEMAKER_TRAIN_CONFIG_PATH) \
  		--param-path $(TRAIN_PARAM_PATH) \
  		--mode AWS \
  		--image-name $(SAGEMAKER_IMAGE_NAME) \
  		--job-id $(JOB_ID)

sagemaker_deploy:
	python ./models/gradient_boost/envs/sagemaker/scripts/deploy.py \
  		--config-path $(SAGEMAKER_DEPLOY_CONFIG_PATH) \
  		--job-id $(JOB_ID)

sagemaker_undeploy:
	python ./models/gradient_boost/envs/sagemaker/scripts/undeploy.py \
		--config-path $(SAGEMAKER_DEPLOY_CONFIG_PATH)