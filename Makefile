build_training_docker:	## Builds training container
	docker build --no-cache -t labsnlp:latest .

run_training_docker: ## Runs training docker
	docker run -it --rm --ipc=host -v $(shell pwd):/workspace labsnlp:latest /bin/bash

run_train_nn_inside_docker: ## Runs training docker
	docker run -it --rm --ipc=host -v $(shell pwd):/workspace labsnlp:latest cd labsnlp && python train_neural_net.py 
## git_sha=$(shell git rev-parse --short HEAD)