# setup
.PHONY: install
install:
	uv sync --no-dev

.PHONY: install-lab
install-lab:
	uv sync --all-packages --no-dev

.PHONY: install-dev
install-dev:
	uv sync --all-packages
	uv run pre-commit install

.PHONY: install-all
install-all: install-dev install-app

# The JAX distribution keeps its own venvs (its CUDA wheels conflict with torch's).
# Create-if-missing rather than --clear: on NFS a venv with files held open (e.g. by
# an IDE's language server) cannot be deleted in place; `rm -rf` it manually if you
# really want a from-scratch env.
.PHONY: install-jax
install-jax:
	cd nano_param_decomp_jax && ([ -x .venv/bin/python ] || \
		( ! [ -e .venv ] || mv .venv .venv-stale-$$(date +%s); \
		  uv venv .venv --python 3.13; rm -rf .venv-stale-* || true )) \
		&& uv pip install -p .venv/bin/python -e ../param_decomp_config -e '.[dev]'

.PHONY: install-jax-cuda
install-jax-cuda:
	cd nano_param_decomp_jax && ([ -x .venv-cuda/bin/python ] || \
		( ! [ -e .venv-cuda ] || mv .venv-cuda .venv-stale-$$(date +%s); \
		  uv venv .venv-cuda --python 3.13; rm -rf .venv-stale-* || true )) \
		&& uv pip install -p .venv-cuda/bin/python -e ../param_decomp_config -e '.[cuda]'

.PHONY: test-jax
test-jax:
	cd nano_param_decomp_jax && .venv/bin/python -m pytest jax_single_pool/tests/

.PHONY: check-jax
check-jax:
	cd nano_param_decomp_jax && .venv/bin/basedpyright jax_single_pool/


.PHONY: app
app:
	@uv run --package param-decomp-lab python -m param_decomp_lab.app.run_app

.PHONY: install-app
install-app:
	(cd param_decomp_lab/app/frontend && npm install)

.PHONY: check-app
check-app:
	(cd param_decomp_lab/app/frontend && npm run format && npm run check && npm run lint)

# special install for CI (GitHub Actions) that reduces disk usage and install time
# 1. create a fresh venv with `--clear` -- this is mostly only for local testing of the CI install
# 2. install with `uv sync` but with some special options:
#  > `--frozen` to enforce using the lock file for consistent dependency versions
#  > `--link-mode copy` because symlinks/hardlinks dont work half the time anyway
#  > `--extra-index-url` to get cpu-only pytorch wheels. installing with just `uv sync` will download a bunch of cuda stuff we cannot use anyway, since there are no GPUs anyways. takes up a lot of space and makes the install take 5x as long
#  > `--index-strategy unsafe-best-match` because pytorch index won't have every version of each package we need. markupsafe is a particular pain point
# Note: explored the `--compile-bytecode` option for test speedups, nothing came of it. see https://github.com/goodfire-ai/param-decomp/pull/187/commits/740f6a28f4d3378078c917125356b6466f155e71
.PHONY: install-ci
install-ci:
	uv venv --python 3.13 --clear
	uv sync \
		--frozen \
		--all-packages \
		--link-mode copy \
		--extra-index-url https://download.pytorch.org/whl/cpu \
		--index-strategy unsafe-best-match

# checks
.PHONY: type
type:
	uv run basedpyright

.PHONY: format
format:
	# Fix all autofixable problems (which sorts imports) then format errors
	uv run ruff check --fix
	uv run ruff format

.PHONY: check
check: format type

.PHONY: check-pre-commit
check-pre-commit:
	SKIP=no-commit-to-branch pre-commit run -a --hook-stage commit

# tests

.PHONY: test
test:
	uv run pytest param_decomp/tests/ param_decomp_lab/tests/ --testmon --durations 10

# Use min(4, nproc) for numprocesses. Any more and it slows down the tests.
NUM_PROCESSES ?= $(shell nproc | awk '{print ($$1<4?$$1:4)}')

.PHONY: test-all
test-all:
	uv run pytest param_decomp/tests/ param_decomp_lab/tests/ --runslow --durations 10 --numprocesses $(NUM_PROCESSES) --dist worksteal

COVERAGE_DIR=docs/coverage

.PHONY: coverage
coverage:
	uv run pytest param_decomp/tests/ param_decomp_lab/tests/ --cov=param_decomp --cov=param_decomp_lab --runslow
	mkdir -p $(COVERAGE_DIR)
	uv run python -m coverage report -m > $(COVERAGE_DIR)/coverage.txt
	uv run python -m coverage html --directory=$(COVERAGE_DIR)/html/


.PHONY: clean
clean:
	@echo "Cleaning Python cache and build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .ruff_cache/ .pytest_cache/ .coverage

