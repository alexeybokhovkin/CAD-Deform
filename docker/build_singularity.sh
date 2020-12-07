#!/bin/bash

set -e

# example launch string:
# ./build_singularity.sh [-f]
#     -f:       overwrite the existing singularity image (false by default)

usage() { echo "Usage: $0 -i alignment|deformation [-f] [-v]" >&2; }

VERBOSE=false
FORCE=false
while getopts "fvi:" opt
do
    case ${opt} in
        f) FORCE=true;;
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


[[ -d ${SIMAGES_DIR} ]] || mkdir ${SIMAGES_DIR}

module load apps/singularity-3.2.0

echo "******* PULLING IMAGE FROM DOCKER HUB AND BUILDING SINGULARITY IMAGE *******"
if [[ "${FORCE}" = true ]] ; then
    singularity pull -F "${SIMAGE_FILENAME}" "docker://${IMAGE_NAME_TAG}"
else
    singularity pull "${SIMAGE_FILENAME}" "docker://${IMAGE_NAME_TAG}"
fi
