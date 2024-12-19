black:
	poetry run black .

isort:
	poetry run isort .

flake8:
	poetry run flake8 --max-line-length=90 --exclude=hfs/lib/,*/__init__.py --extend-ignore=E741,W503,W605,E501 hfs/

autolint: isort black flake8

mypy:
	poetry run mypy hfs/

pytest:
	poetry run pytest hfs/

pytest-until-fail:
	poetry run pytest hfs/ -x

run-all: autolint pytest
