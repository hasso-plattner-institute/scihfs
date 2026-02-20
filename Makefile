black:
	uv run black .

isort:
	uv run isort .

flake8:
	uv run flake8 --max-line-length=90 --exclude=scihfs/lib/,*/__init__.py --extend-ignore=E741,W503,W605,E501 scihfs/

autolint: isort black flake8

mypy:
	uv run mypy scihfs/

pytest:
	uv run pytest scihfs/

pytest-until-fail:
	uv run pytest scihfs/ -x

run-all: autolint pytest

build-and-publish:
	rm -rf dist/
	poetry build
	poetry run twine upload --repository testpypi --skip-existing dist/*
