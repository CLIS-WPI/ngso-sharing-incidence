# All targets run inside the dev container (GPU 0 only).
COMPOSE := docker compose -f docker/docker-compose.yml
export USER_UID := $(shell id -u)
export USER_GID := $(shell id -g)

.PHONY: help precheck build up down shell test run-all regenerate clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

precheck: ## Verify host GPU + NVIDIA container toolkit
	nvidia-smi
	@docker info | grep -i nvidia

build: ## Build the dev image
	$(COMPOSE) build

up: ## Start the dev container (detached)
	$(COMPOSE) up -d

down: ## Stop and remove the dev container
	$(COMPOSE) down

shell: up ## Open a shell in the dev container
	$(COMPOSE) exec dev bash

test: up ## Run the test suite
	$(COMPOSE) exec dev python -m pytest -q

run-all: up ## Re-run every experiment, then numbers and figures
	$(COMPOSE) exec dev bash scripts/run_all.sh

regenerate: up ## output/numbers.tex and output/figures from results/ (no GPU)
	$(COMPOSE) exec dev python scripts/make_numbers.py
	$(COMPOSE) exec dev python figures/make_figures.py

clean: ## Remove the dev container and image
	$(COMPOSE) down --rmi local --volumes
