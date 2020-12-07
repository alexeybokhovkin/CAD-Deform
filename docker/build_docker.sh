#!/bin/bash

set -e

# example launch string:
# ./build_docker.sh [-p] [-v]
#     -p:       push the build image to the dockerhub under 'artonson' username
#     -v:       be verbose

usage() { echo "Usage: $0 -i alignment|deformation [-p] [-v]" >&2; }

VERBOSE=false
PUSH=false
while getopts "pvi:" opt
do
    case ${opt} in
        p) PUSH=true;;
        v) VERBOSE=true;;
        i) DOCKER_TYPE=${OPTARG};;
        *) usage; exit 1 ;;
    esac
done

if [[ "${VERBOSE}" = true ]]; then
    set -x
fi

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null 2>&1 && pwd )"
source "${PROJECT_ROOT}"/env.sh

if [[ ! ${DOCKER_TYPE} ]]; then
    echo "docker_type is not set" && usage && exit 1
else
    if [[ "${DOCKER_TYPE}" != "${IMAGE_VERSION_ALIGNMENT}"  && "${DOCKER_TYPE}" != "${IMAGE_VERSION_DEFORMATION}" ]] ; then
      echo "docker_type: select 'alignment' or 'deformation'" && usage && exit 1
    fi
  # manually set image version/tag
    IMAGE_VERSION=${DOCKER_TYPE}
    IMAGE_NAME_TAG="${IMAGE_NAME}:${IMAGE_VERSION}"
    SIMAGE_FILENAME="${SIMAGES_DIR}/$(echo ${IMAGE_NAME_TAG} | tr /: _).sif"
fi


DOCKERFILE="${PROJECT_ROOT}/docker/${DOCKER_TYPE}/Dockerfile"     # full pathname of Dockerfile

echo "******* BUILDING IMAGE ${IMAGE_NAME} *******"

docker build \
    --file "${DOCKERFILE}" \
    --tag "${IMAGE_NAME_TAG}" \
    "${PROJECT_ROOT}"


if [ "${PUSH}" = true ]; then
    echo "******* LOGGING TO DOCKER HUB *******"
    docker login

    echo "******* PUSHING IMAGE TO DOCKER HUB *******"
    docker push "${IMAGE_NAME_TAG}"
fi
