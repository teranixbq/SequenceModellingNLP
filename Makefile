venv:
	python3.11 -m venv venv

install:
	pip install -r requirements.txt

frz:
	pip freeze > requirements.txt