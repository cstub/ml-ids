#!/usr/bin/env bash

image_name=$1
image_version=$2

if [ "$image_name" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

if [ "$image_version" == "" ]
then
    echo "Usage: $1 <image-version>"
    exit 1
fi

fullname="${image_name}:${image_version}"

echo "Building image '${fullname}'"

docker build -f models/gradient_boost/envs/sagemaker/container/Dockerfile -t ${fullname} .