code-style-check:
	black irec/environment/ --check
	isort irec/environment/ --check
	flake8

code-style-format:
	black irec/environment/
	isort irec/environment/