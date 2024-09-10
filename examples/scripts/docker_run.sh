set -x

build=${1-""}
PROJECT_PATH=$(cd $(dirname $0)/../../; pwd)
IMAGE_NAME="nvcr.io/nvidia/pytorch:23.12-py3"

if [[ "${build}" == *"b"* ]]; then
	docker image rm $IMAGE_NAME
	docker build -t $IMAGE_NAME $PROJECT_PATH/dockerfile 
else 
	docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
		-v $PROJECT_PATH:/openrlhf -v  $HOME/.cache:/root/.cache -v  $HOME/.bash_history2:/root/.bash_history \
		$IMAGE_NAME bash
fi