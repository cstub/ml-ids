## Data

The data used to train the classifiers is taken from the [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html) dataset provided by the Canadian Institute for Cybersecurity.
It was created by capturing all network traffic during ten days of operation inside a controlled network environment on AWS where realistic background traffic and different attack scenarios were conducted.

The dataset consists of raw network captures in pcap format as well as processed csv files created by using [CICFlowMeter-V3](https://www.unb.ca/cic/research/applications.html#CICFlowMeter) containing 80 statistical features of the individual network flows combined with their corresponding labels.

Due to size limitations the data provided in this repository represents only a small portion of the dataset in form of processed network flows. The full dataset consisting of the raw network captures and the processed csv files can be retrieved from AWS S3.

## Download

A prerequisite to downloading the full dataset is the installation of the [AWS CLI](https://aws.amazon.com/cli/).

To download the processed csv files containing the analyzed network flows (~7GB) run the following command:
```bash
aws s3 sync --no-sign-request --region <your-region> "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" <dest-dir>
```
To download the raw network captures in pcap format (~477GB) run:
```bash
aws s3 sync --no-sign-request --region <your-region> "s3://cse-cic-ids2018/Original Network Traffic and Log data/" <dest-dir>
```
To download the full dataset containing the raw network captures and processed csv files (~484GB) use the following command:
```bash
aws s3 sync --no-sign-request --region <your-region> "s3://cse-cic-ids2018/" <dest-dir>
```

## Preprocessed Dataset

The preprocessed dataset used for model training and evaluation can be found at [Google Drive](https://drive.google.com/drive/folders/1AWhRsVShJ_KvYKrV0VlnM1odtJ4Tp-uC?usp=sharing).
