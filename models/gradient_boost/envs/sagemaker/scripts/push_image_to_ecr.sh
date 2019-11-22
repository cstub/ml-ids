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

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi

# Get the region defined in the current configuration (default to eu-west-1 if none defined)
region=$(aws configure get region)
region=${region:-eu-west-1}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image_name}:${image_version}"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${image_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image_name}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker tag "${image_name}:${image_version}" ${fullname}
docker push ${fullname}

echo "image-name=${fullname}"