# Contributing

Any contributions to 123D are welcome! This guide both serves as internal tutorial and can help you get started with the development process.

## Table of contents

1. [Ways to contribute](#1-ways-to-contribute)
2. [Installation](#2-installation)
3. [Conventions](#3-conventions)
4. [Dependencies](#4-dependencies)
5. [Testing](#5-testing)
6. [Submitting a pull-request](#6-submitting-a-pull-request)


## 1. Ways to contribute

If you want to get involved and improve 123D, there are several ways to contribute, that include but are not limited to:

- **Support new datasets:** Our framework initially focussed on the most common driving datasets available. Since more datasets are released each year, we hope to extend the support throughout. If you consider contributing dataset support, please read our designated tutorial: [Adding Datasets](https://kesai.eu/py123d/notes/adding_datasets)

- **Improve documentation:** Anything unclear or not properly documented? Feel free to reach out or create a pull-request, so that others can benefit.

- **Adding or extend features:** Features and tools may need to be improved in terms of performance, coverage, or scope.


## 2. Installation

You can get started by
```sh
git clone git@github.com:kesai-labs/py123d.git
cd py123d
```

and install all development dependencies with
```sh
conda create -n py123d_dev python=3.12; conda activate py123d_dev  # Optional
pip install uv
uv pip install -e .[dev,docs]
pre-commit install
```

`pre-commit install` registers the hooks so that linting and formatting run automatically on every `git commit`.
We use [`ruff`](https://docs.astral.sh/ruff/) as linter/formatter, which you can also run manually:
```sh
ruff check --fix .
ruff format .
```
Type checking is not strictly enforced, but ideally added with [`pyright`](https://github.com/microsoft/pyright).

## 3. Conventions

### Code style
We use [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting, enforced via `pre-commit` and CI (see [Installation](#2-installation)). The rules live in `pyproject.toml`:
- **Line length** is 120 characters.
- **Imports** are sorted automatically (isort, `black` profile): standard library, third-party, then `py123d`.

### Type hints
Annotate function arguments and return types. Use [`numpy.typing`](https://numpy.org/doc/stable/reference/typing.html) (e.g. `npt.NDArray[np.float64]`) for array types. Type checking with [`pyright`](https://github.com/microsoft/pyright) is encouraged but not enforced.

### Docstrings
Docstrings are encouraged but not strictly required during active development. Use [Sphinx-style](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html) docstrings with `:param:` / `:return:` fields:

```python
def get_by_lane_id(self, lane_id: int) -> Optional[TrafficLightDetection]:
    """Retrieve a traffic light detection by its lane id.

    :param lane_id: The lane id to search for.
    :return: The detection for the given lane id, or None if not found.
    """
```

### Naming
- `snake_case` for modules, functions, and variables; a leading `_` marks private members.
- `PascalCase` for classes; `UPPER_SNAKE_CASE` for constants and enum values.
- Abstract base classes are prefixed with `Base` (e.g. `BaseDatasetParser`); dataset parsers and metadata containers use the `<Dataset>Parser` and `<...>Metadata` suffixes.
- Re-export a subpackage's public API from its `__init__.py` via `__all__` (these files are excluded from `ruff`).

### Domain conventions
Coordinate system, poses (SE(2)/SE(3)), quaternions, and bounding-box layouts follow a single unified representation documented in [Conventions](https://kesai.eu/py123d/notes/conventions).

### Sphinx documentation
All datasets should be documented under `docs/datasets/`; follow the format of the existing datasets. Install the docs dependencies and preview live with:

```sh
pip install -e .[docs]
sphinx-autobuild docs docs/_build/html
```


## 4. Dependencies

We try to keep dependencies minimal to ensure quick and easy installations.
However, dataset specific code may require dependencies in order to load or preprocess the dataset.
In this case, you can add optional dependencies to the `pyproject.toml` install file.
You can follow examples of nuPlan or nuScenes. These optional dependencies can be installed with

```sh
pip install -e .[dev,nuplan,nuscenes]
```
where you can combine the different optional dependencies.

Please make sure that optional dependencies remain dataset specific and affect the code in `src/py123d/parser/<...>`.
When writing a dataset conversion method, you can check if the necessary dependencies are installed by calling with the `check_dependencies` function.

```python
from py123d.common.utils.dependencies import check_dependencies

check_dependencies(["optional_package_a", "optional_package_b"], "optional_dataset")
import optional_package_a
import optional_package_b

def load_camera_from_outdated_dataset(...) -> ...:
    optional_package_a.module(...)
    optional_package_b.module(...)
    pass
```
This will notify the user if `optional_dataset` is not yet installed and how to resolve the dependency issue.

Also ensure that functions/modules that require optional installs are only imported when necessary, e.g:

```python
def load_camera_from_file(file_path: str, dataset: str) -> ...:
    ...
    if dataset == "optional_dataset":
        from py123d.some_module import load_camera_from_outdated_dataset

        return load_camera_from_outdated_dataset(...)
    ...
```

## 5. Testing

We use [`pytest`](https://docs.pytest.org/). Tests live in `tests/`, mirroring the package layout:
- `tests/unit/`: unit tests for `src/py123d/` (e.g. `tests/unit/geometry/test_pose.py`).
- `tests/docs/`: verifies that the documentation builds.

Name test files `test_<module>.py`, group related tests in `class Test<Subject>:`, and name methods `test_<behavior>`. Shared fixtures live in a `conftest.py` next to the tests that use them.

Run the suite locally:

```sh
pytest tests/unit                 # unit tests (slow tests excluded by default)
pytest tests/unit -m ''           # also run @pytest.mark.slow tests
pytest tests/unit --cov           # with coverage
pytest tests/unit/geometry/test_pose.py::TestPoseSE2   # a single file/class/test
```

CI runs `pytest tests/unit` on Python 3.9–3.13 and `pytest tests/docs` (Python 3.11) for every pull request and push to `main`. Please add tests for new features and bug fixes.

Tests for dataset-specific code in `src/py123d/parser/<...>` require optional dependencies and access to the dataset files, so they are **not** run in CI. Mark long-running tests with `@pytest.mark.slow`, and skip cleanly when an optional dependency is missing (see [`check_dependencies`](#4-dependencies)).


## 6. Submitting a pull-request

Development happens on a versioned branch named `dev_vX.Y.Z`. The latest one is merged into `main` at each release. Please target the **current development branch**, not `main`.

1. Branch off the current `dev_v*` branch (external contributors: fork first). Name the branch by type, e.g. `feat/radar-support`, `fix/msgpack-keys`, or `docs/update-docs`.
2. Make focused commits following [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `chore:`, `docs:`, or `refactor:`.
3. Before opening the PR, make sure the checks pass locally:
   ```sh
   pre-commit run --all-files     # ruff lint + format and file checks
   pytest tests/unit              # add `pytest tests/docs` if you touched the docs
   ```
4. Open the PR against the current `dev_v*` branch. Describe what changed and why, and link any related issues.

Your PR must pass the `pre-commit` and `pytest` GitHub Actions (linting/formatting plus the unit and docs tests across Python 3.9–3.13). Contributions are accepted under the project's [Apache-2.0 license](https://github.com/kesai-labs/py123d/blob/main/LICENSE).
