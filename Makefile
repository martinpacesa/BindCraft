# BindCraft Docker Makefile

.PHONY: help build build-prod up down logs clean test docs

# Variables
DOCKER_IMAGE := bindcraft:latest
DOCKER_REGISTRY := localhost
COMPOSE := docker-compose
COMPOSE_FILE := docker-compose.yml

help:
	@echo "BindCraft Docker - Available commands:"
	@echo ""
	@echo "Build & Run:"
	@echo "  make build              Build Docker image"
	@echo "  make build-prod         Build optimized production image"
	@echo "  make up                 Start API server + worker (GPU 0 & 1)"
	@echo "  make down               Stop all services"
	@echo "  make restart            Restart services"
	@echo ""
	@echo "Development:"
	@echo "  make logs               Follow API logs"
	@echo "  make shell              Open shell in API container"
	@echo "  make test               Run tests"
	@echo "  make bench              Run performance benchmarks"
	@echo ""
	@echo "Deployment:"
	@echo "  make push               Push image to registry"
	@echo "  make docs               Generate API documentation"
	@echo "  make clean              Clean build artifacts"
	@echo ""

# Build targets
build:
	@echo "🔨 Building BindCraft Docker image..."
	docker build -f Dockerfile.bindcraft -t $(DOCKER_IMAGE) .
	@echo "✅ Build complete: $(DOCKER_IMAGE)"

build-prod:
	@echo "🔨 Building production image (multi-stage optimization)..."
	docker build \
		--target production \
		-f Dockerfile.bindcraft \
		-t $(DOCKER_IMAGE)-prod \
		--build-arg CUDA_VERSION=12.4 \
		.
	@echo "✅ Production build complete"

# Compose operations
up:
	@echo "🚀 Starting BindCraft services..."
	$(COMPOSE) -f $(COMPOSE_FILE) up -d
	@echo "✅ Services started"
	@echo "API: http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"

down:
	@echo "⏹️ Stopping services..."
	$(COMPOSE) -f $(COMPOSE_FILE) down
	@echo "✅ Services stopped"

restart:
	@echo "🔄 Restarting services..."
	$(COMPOSE) -f $(COMPOSE_FILE) restart
	@echo "✅ Services restarted"

logs:
	@echo "📋 Following API logs..."
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f bindcraft-api

logs-worker:
	@echo "📋 Following worker logs..."
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f bindcraft-worker

# Development
shell:
	@echo "🔧 Opening shell in API container..."
	docker exec -it bindcraft-api bash

shell-worker:
	@echo "🔧 Opening shell in worker container..."
	docker exec -it bindcraft-worker bash

test:
	@echo "🧪 Running tests..."
	docker run --rm \
		--gpus all \
		-v $(PWD)/tests:/workspace/tests \
		-v $(PWD)/data:/data \
		$(DOCKER_IMAGE) \
		bash -c "cd /workspace && python -m pytest tests/ -v"
	@echo "✅ Tests complete"

bench:
	@echo "⚡ Running performance benchmarks..."
	docker run --rm \
		--gpus all \
		-v $(PWD)/data:/data \
		$(DOCKER_IMAGE) \
		python /workspace/docker/benchmark.py
	@echo "✅ Benchmarks complete"

# Deployment
push:
	@echo "📤 Pushing image to registry..."
	docker tag $(DOCKER_IMAGE) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE)
	@echo "✅ Image pushed"

docs:
	@echo "📚 Generating API documentation..."
	docker run --rm \
		-v $(PWD)/docs:/docs \
		$(DOCKER_IMAGE) \
		bash -c "python -m mkdocs build"
	@echo "✅ Documentation generated in /docs"

# Utilities
clean:
	@echo "🧹 Cleaning up..."
	$(COMPOSE) -f $(COMPOSE_FILE) down -v
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	docker system prune -f
	@echo "✅ Cleanup complete"

ps:
	@$(COMPOSE) -f $(COMPOSE_FILE) ps

ps-verbose:
	@docker ps --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"

# GPU utilities
gpu-stats:
	@echo "🖥️  GPU Status:"
	@nvidia-smi

gpu-monitor:
	@watch -n 1 nvidia-smi

# Quick start
quickstart: build up
	@echo ""
	@echo "🚀 BindCraft is running!"
	@echo "API Endpoint: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"
	@echo ""
	@echo "Try:"
	@echo "  curl http://localhost:8000/info"
	@echo "  curl http://localhost:8000/health"

# Integration test
integration-test:
	@echo "🧪 Running integration tests..."
	@echo "1. Uploading test PDB..."
	@curl -X POST -F "file=@tests/fixtures/test_target.pdb" http://localhost:8000/upload
	@echo ""
	@echo "2. Submitting design job..."
	@curl -X POST http://localhost:8000/design \
		-H "Content-Type: application/json" \
		-d '{"target_pdb_file": "test_target.pdb", "binder_name": "test", "num_designs": 10}'
	@echo ""
	@echo "✅ Integration test complete"

# Info
info:
	@echo "BindCraft Docker Configuration:"
	@echo "  Image: $(DOCKER_IMAGE)"
	@echo "  Registry: $(DOCKER_REGISTRY)"
	@echo "  Compose: $(COMPOSE_FILE)"
	@echo ""
	@echo "Active Containers:"
	@$(COMPOSE) -f $(COMPOSE_FILE) ps --services || echo "  None (run 'make up')"
	@echo ""

.DEFAULT_GOAL := help
