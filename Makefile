.PHONY: install data train test run clean lint

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && pip install -e .

data:
	python -m zakkend.data.synthetic --n 10000 --out data/processed/training.parquet

train:
	python scripts/train.py

test:
	pytest -v

run:
	uvicorn zakkend.api.main:app --reload --host 0.0.0.0 --port 8000

lint:
	ruff check src tests
	black --check src tests

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
