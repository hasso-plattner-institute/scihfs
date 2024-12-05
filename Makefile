black:
	poetry run black .

isort:
	poetry run isort .

autolint: isort black

mypy:
	poetry run mypy hfs/

pytest:
	poetry run pytest hfs

run-all: autolint pytest
