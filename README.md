# A machine learning based approach towards building an Intrusion Detection System

## Problem Description
With the rising amount of network enabled devices connected to the internet such as mobile phones, IOT appliances or vehicles the concern about the security implications of using these devices is growing. The increase in numbers and types of networked devices inevitably leads to a wider surface of attack whereas the impact of successful attacks is becoming increasingly severe as more critical responsibilities are assumed be these devices.

To identify and counter network attacks it is common to employ a combination of multiple systems in order to prevent attacks from happening or to detect and stop ongoing attacks if they can not be prevented initially.
These systems are usually comprised of an intrusion prevention system such as a firewall as the first layer of security with intrusion detection systems representing the second layer.
Should the intrusion prevention system be unable to prevent a network attack it is the task of the detection system to identify malicious network traffic in order to stop the ongoing attack and keep the recorded network traffic data for later analysis. This data can subsequently be used to update the prevention system to allow for the detection of the specific network attack in the future. The need for intrusion detection systems is rising as absolute prevention against attacks is not possible due to the rapid emergence of new attack types.

Even though intrusion detection systems are an essential part of network security many detection systems deployed today have a significant weakness as they facilitate signature-based attack classification patterns which are able to detect the most common known attack patterns but have the drawback of being unable to detect novel attack types.
To overcome this limitation research in intrusion detection systems is focusing on more dynamic approaches based on machine learning and anomaly detection methods. In these systems the normal network behaviour is learned by processing previously recorded benign data packets which allows the system to identify new attack types by analyzing network traffic for anomalous data flows.

This project aims to implement a classifier capable of identifying network traffic as either benign or malicious based on machine learning and deep learning methodologies.

## Data
The data used to train the classifier is taken from the [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html) dataset provided by the Canadian Institute for Cybersecurity. It was created by capturing all network traffic during ten days of operation inside a controlled network environment on AWS where realistic background traffic and different attack scenarios were conducted.
As a result the dataset contains both benign network traffic as well as captures of the most common network attacks.
The dataset is comprised of the raw network captures in pcap format as well as csv files created by using [CICFlowMeter-V3](https://www.unb.ca/cic/research/applications.html#CICFlowMeter) containing 80 statistical features of the individual network flows combined with their corresponding labels.
A network flow is defined as an aggregation of interrelated network packets identified by the following properties:
* Source IP
* Destination IP
* Source port
* Destination port
* Protocol

The dataset contains approximately 16 million individual network flows and covers the following attack scenarios:
* Brute Force
* DoS,
* DDos
* Heartbleed,
* Web Attack,
* Infiltration,
* Botnet

## Approach
The goal of this project is to create a classifier capable of categorising network flows as either benign or malicious.
The problem is understood as a supervised learning problem using the labels provided in the dataset which identify the network flows as either benign or malicious. Different approaches of classifying the data will be evaluated to formulate the problem either as a binary classification or a multiclass classification problem differentiating between the individual classes of attacks provided in the dataset in the later case. A relevant subset of the features provided in the dataset will be used as predictors to classify individual network flows.
Machine learning methods like k-nearest neighbours, random forest or SVM will be applied to the problem and evaluated in the first step in order to assess the feasibility of using traditional machine learning approaches.
Subsequently deep learning models like convolutional neural networks, autoencoders or recurrent neural networks will be employed to create a competing classifier as recent research has shown that deep learning methods represent a promising application in the field of anomaly detection.
The results of both approaches will be compared to select the best performing classifier.

## Deliverables
The classifier will be deployed and served via a REST API in conjunction with a simple web application providing a user interface to utilize the API.

The REST API will provide the following functionality:
* an endpoint to submit network capture files in pcap format. Individual network flows are extracted from the capture files and analysed for malicious network traffic.
* (optional) an endpoint to stream continuous network traffic captures which are analysed in near real-time combined with
* (optional) an endpoint to register a web-socket in order to get notified upon detection of malicious network traffic.

To further showcase the project, a testbed could be created against which various attack scenarios can be performed. This testbed would be connected to the streaming API for near real-time detection of malicious network traffic.

## Computational resources
The requirements regarding the computational resources to train the classifiers are given below:

| Category      | Resource      |
| ------------- | ------------- |
| CPU | Intel Core i7 processor |
| RAM | 32 GB                   |
| GPU | 1 GPU, 8 GB RAM         |
| HDD | 100 GB                  |


## Classifier

The machine learning estimator created in this project follows a supervised approach and is trained using the [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) algorithm. Employing the [CatBoost](https://catboost.ai/) library a binary classifier is created, capable of classifying network flows as either benign or malicious. The chosen parameters of the classifier and its performance metrics can be examined in the following [notebook](https://github.com/cstub/ml-ids/blob/master/notebooks/07_binary_classifier_comparison/binary-classifier-comparison.ipynb).     

## Deployment Architecture

The deployment architecture of the complete ML-IDS system is explained in detail in the [system architecture](https://docs.google.com/document/d/1s_EBMTid4gdrsQU_xOCAYK1BzxkhhnYl6wHFSZo_9Tw/edit?usp=sharing).

## Model Training and Deployment

The model can be trained and deployed either locally or via [Amazon SageMaker](https://aws.amazon.com/sagemaker/).     
In each case the [MLflow](https://www.mlflow.org/docs/latest/index.html) framework is utilized to train the model and create the model artifacts.

### Installation

To install the necessary dependencies checkout the project and create a new Anaconda environment from the environment.yml file.

```
conda env create -f environment.yml
```

Afterwards activate the environment and install the project resources.

```
conda activate ml-ids

pip install -e .
```

### Dataset Creation

To create the dataset for training use the following command:

```
make split_dataset \
  DATASET_PATH={path-to-source-dataset}
```

This command will read the source dataset and split the dataset into separate train/validation/test sets with a sample ratio of 80%/10%/10%. The specified source dataset should be a folder containing multiple `.csv` files.    
You can use the [CIC-IDS-2018 dataset](https://www.unb.ca/cic/datasets/ids-2018.html) provided via [Google Drive](https://drive.google.com/open?id=1HrTPh0YRSZ4T9DLa_c47lubheKUcPl0r) for this purpose.    
Once the command completes a new folder `dataset` is created that contains the splitted datasets in `.h5` format.

### Local Mode

To train the model in local mode, using the default parameters and dataset locations created by `split_dataset`, use the following command:

```
make train_local
```

If the datasets are stored in a different location or you want to specify different training parameters, you can optionally supply the dataset locations and a training parameter file:

```
make train_local \
  TRAIN_PATH={path-to-train-dataset} \
  VAL_PATH={path-to-train-dataset} \
  TEST_PATH={path-to-train-dataset} \
  TRAIN_PARAM_PATH={path-to-param-file}
```

Upon completion of the training process the model artifacts can be found in the `build/models/gradient_boost` directory.

To deploy the model locally the MLflow CLI can be used.

```
mlflow models serve -m build/models/gradient_boost -p 5000
```

The model can also be deployed as a Docker container using the following commands:

```
mlflow models build-docker -m build/models/gradient_boost -n ml-ids-classifier:1.0

docker run -p 5001:8080 ml-ids-classifier:1.0
```

### Amazon SageMaker

To train the model on Amazon SageMaker the following command sequence is used:

```
# build a new docker container for model training
make sagemaker_build_image \
  TAG=1.0

# upload the container to AWS ECR
make sagemaker_push_image \
  TAG=1.0

# execute the training container on Amazon SageMaker
make sagemaker_train_aws \
  SAGEMAKER_IMAGE_NAME={ecr-image-name}:1.0 \
  JOB_ID=ml-ids-job-0001
```

This command requires a valid AWS account with the appropriate permissions to be configured locally via the [AWS CLI](https://aws.amazon.com/cli/). Furthermore, [AWS ECR](https://aws.amazon.com/ecr/) and Amazon SageMaker must be configured for the account.

Using this repository, the manual invocation of the aforementioned commands is not necessary as training on Amazon SageMaker is supported via a [GitHub workflow](https://github.com/cstub/ml-ids/blob/master/.github/workflows/train.yml) that is triggered upon creation of a new tag of the form `m*` (e.g. `m1.0`).

To deploy a trained model on Amazon SageMaker a [GitHub Deployment request](https://developer.github.com/v3/repos/deployments/) using the GitHub API must be issued, specifying the tag of the model.

```
{
  "ref": "refs/tags/m1.0",
  "payload": {},
  "description": "Deploy request for model version m1.0",
  "auto_merge": false
}
```

This deployment request triggers a [GitHub workflow](https://github.com/cstub/ml-ids/blob/master/.github/workflows/deployment.yml), deploying the model to SageMaker.
After successful deployment the model is accessible via the SageMaker HTTP API.

## Using the Classifier

The classifier deployed on Amazon SageMaker is not directly available publicly, but can be accessed using the [ML-IDS REST API](https://github.com/cstub/ml-ids-api).  

### REST API

To invoke the REST API the following command can be used to submit a prediction request for a given network flow:

```
curl -X POST \
  http://ml-ids-cluster-lb-1096011980.eu-west-1.elb.amazonaws.com/api/predictions \
  -H 'Accept: */*' \
  -H 'Content-Type: application/json; format=pandas-split' \
  -H 'Host: ml-ids-cluster-lb-1096011980.eu-west-1.elb.amazonaws.com' \
  -H 'cache-control: no-cache' \
  -d '{"columns":["dst_port","protocol","timestamp","flow_duration","tot_fwd_pkts","tot_bwd_pkts","totlen_fwd_pkts","totlen_bwd_pkts","fwd_pkt_len_max","fwd_pkt_len_min","fwd_pkt_len_mean","fwd_pkt_len_std","bwd_pkt_len_max","bwd_pkt_len_min","bwd_pkt_len_mean","bwd_pkt_len_std","flow_byts_s","flow_pkts_s","flow_iat_mean","flow_iat_std","flow_iat_max","flow_iat_min","fwd_iat_tot","fwd_iat_mean","fwd_iat_std","fwd_iat_max","fwd_iat_min","bwd_iat_tot","bwd_iat_mean","bwd_iat_std","bwd_iat_max","bwd_iat_min","fwd_psh_flags","bwd_psh_flags","fwd_urg_flags","bwd_urg_flags","fwd_header_len","bwd_header_len","fwd_pkts_s","bwd_pkts_s","pkt_len_min","pkt_len_max","pkt_len_mean","pkt_len_std","pkt_len_var","fin_flag_cnt","syn_flag_cnt","rst_flag_cnt","psh_flag_cnt","ack_flag_cnt","urg_flag_cnt","cwe_flag_count","ece_flag_cnt","down_up_ratio","pkt_size_avg","fwd_seg_size_avg","bwd_seg_size_avg","fwd_byts_b_avg","fwd_pkts_b_avg","fwd_blk_rate_avg","bwd_byts_b_avg","bwd_pkts_b_avg","bwd_blk_rate_avg","subflow_fwd_pkts","subflow_fwd_byts","subflow_bwd_pkts","subflow_bwd_byts","init_fwd_win_byts","init_bwd_win_byts","fwd_act_data_pkts","fwd_seg_size_min","active_mean","active_std","active_max","active_min","idle_mean","idle_std","idle_max","idle_min"],"data":[[80,17,"21\\/02\\/2018 10:15:06",119759145,75837,0,2426784,0,32,32,32.0,0.0,0,0,0.0,0.0,20263.87212,633.2460039,1579.1859130859,31767.046875,920247,1,120000000,1579.1859130859,31767.046875,920247,1,0,0.0,0.0,0,0,0,0,0,0,606696,0,633.2460327148,0.0,32,32,32.0,0.0,0.0,0,0,0,0,0,0,0,0,0,32.0004234314,32.0,0.0,0,0,0,0,0,0,75837,2426784,0,0,-1,-1,75836,8,0.0,0.0,0,0,0.0,0.0,0,0]]}'
```

### ML-IDS API Clients

For convenience, the Python clients implemented in the [ML-IDS API Clients project](https://github.com/cstub/ml-ids-api-client) can be used to submit new prediction requests to the API and receive real-time notifications on detection of malicious network flows.
