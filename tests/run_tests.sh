#!/usr/bin/env bash

# Define some variables to use in this script
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
TEST_RESOURCES_DIR="${SCRIPT_DIR}/resources/pretrained/Task004_Hippocampus"
CODEBASE_DIR=`realpath "${SCRIPT_DIR}/../"`
DOCKER_DEV_IMAGE="nnunet_dev_image:latest"

# Display help and warnings
echo "!!! Warning: This script generates a Docker image and downloads the required test files if not present."
echo "!!! This script assumes you are running from a Linux machine with docker installed."
echo "!!! The required test files (that will be downloaded) take approximately 300MB of disk space and will be downloaded to: "
echo "!!! ${TEST_RESOURCES_DIR}"
echo "!!! The required Docker image can take up to 11GB and will be given the tag: ${DOCKER_DEV_IMAGE}"

# Test if test environment docker image is present, otherwise build it
if [[ "`docker images -q nnunet_dev_image:latest`" == "" ]]; then
  echo "# no '${DOCKER_DEV_IMAGE}' docker image found, building..."
  docker build "${SCRIPT_DIR}/docker" --tag="${DOCKER_DEV_IMAGE}"
else
  echo "# Reusing '${DOCKER_DEV_IMAGE}' docker image..."
fi
if [[ "`docker images -q ${DOCKER_DEV_IMAGE}`" == "" ]]; then
  echo "# No '${DOCKER_DEV_IMAGE}' docker image found after build step, something went wrong, aborting now..."
  exit 1
fi

# Test if all required test files are present, otherwise download them...
if [ ! -d "${TEST_RESOURCES_DIR}" ]; then
  echo "# Test resources are missing (no directory: ${TEST_RESOURCES_DIR} found), downloading them now..."
  docker run -it --rm -v "${CODEBASE_DIR}:/codebase" -w "/codebase" ${DOCKER_DEV_IMAGE} python3.8 -c "
import nnunet.inference.pretrained_models.download_pretrained_model;
from nnunet.inference.pretrained_models.download_pretrained_model import download_and_install_pretrained_model_by_name;
nnunet.inference.pretrained_models.download_pretrained_model.network_training_output_dir = str('/codebase/tests/resources/pretrained/Task004_Hippocampus');
download_and_install_pretrained_model_by_name(taskname='Task004_Hippocampus');"
else
  echo "# Test resources are present under: ${TEST_RESOURCES_DIR}..."
fi

# Run all tests
echo "# Running all tests using docker image: '${DOCKER_DEV_IMAGE}'"
docker run -it --rm --runtime=nvidia -v "${CODEBASE_DIR}:/codebase" -w "/codebase" ${DOCKER_DEV_IMAGE} python3.8 -m pytest "/codebase/tests"
