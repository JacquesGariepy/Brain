.PHONY: help install install-dev clean test lint format docker-build docker-run docs

help:
	@echo "Commandes disponibles:"
	@echo "  make install       - Installer les dépendances de production"
	@echo "  make install-dev   - Installer les dépendances de développement"
	@echo "  make clean         - Nettoyer les fichiers temporaires"
	@echo "  make test          - Exécuter les tests"
	@echo "  make lint          - Vérifier la qualité du code"
	@echo "  make format        - Formater le code"
	@echo "  make docker-build  - Construire l'image Docker"
	@echo "  make docker-run    - Lancer le conteneur Docker"
	@echo "  make docs          - Générer la documentation"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	rm -rf build dist .pytest_cache .coverage htmlcov
	rm -rf logs/*.log
	rm -f long_term_memory.json

test:
	pytest

test-verbose:
	pytest -vv

test-coverage:
	pytest --cov=. --cov-report=html --cov-report=term

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	pylint core modules utils

format:
	black .
	isort .

docker-build:
	docker build -t brain-neural-network:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f brain

docs:
	cd docs && make html

run:
	python main.py

run-demo:
	python main.py --demo

run-interactive:
	python main.py --interactive
