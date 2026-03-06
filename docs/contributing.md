# Contributing

Contributions to 123D are highly encouraged! This guide will help you get started with the development process.

## Getting Started

### 1. Clone the Repository

```sh
git clone git@github.com:autonomousvision/py123d.git
cd py123d
```

### 2. Installation

```sh
conda create -n py123d_dev python=3.12  # Optional
conda activate py123d_dev
pip install -e .[dev]
pre-commit install
```

The above installation should also include linting, formatting, type-checking in the pre-commit.
We use [`ruff`](https://docs.astral.sh/ruff/) as linter/formatter, for which you can run:
```sh
ruff check --fix .
ruff format .
```
Type checking is not strictly enforced, but ideally added with [`pyright`](https://github.com/microsoft/pyright).


### 3. Managing dependencies

We try to keep dependencies minimal to ensure quick and easy installations.
However, various datasets require dependencies in order to load or preprocess the dataset.
In this case, you can add optional dependencies to the `pyproject.toml` install file.
You can follow examples of nuPlan or nuScenes. These optional dependencies can be install with

```sh
pip install -e .[dev,nuplan,nuscenes]
```
where you can combined the different optional dependencies.

The optional dependencies should only be required for data pre-processing.
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
This will notify the user if `optional_dataset` is not included in the 123D install.

Also ensure that functions/modules that require optional installs are only imported when necessary, e.g:

```python
def load_camera_from_file(file_path: str, dataset: str) -> ...:
    ...
    if dataset == "optional_dataset":
        from py123d.some_module import load_camera_from_outdated_dataset

        return load_camera_from_outdated_dataset(...)
    ...
```

### 4. Other useful tools

If you are using VSCode, it is recommended to install:
- [autodocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) - Creating docstrings (please set `"autoDocstring.docstringFormat": "sphinx-notypes"`).
- [Code Spell Checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker) - A basic spell checker.

Or other similar plugins depending on your preference/editor.

## Documentation Requirements

### Docstrings
- **Development:** Docstrings are encouraged but not strictly required during active development
- **Format:** Use [Sphinx-style docstrings](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html)


### Sphinx documentation

All datasets should be included in the `/docs/datasets/` documentation. Please follow the documentation format of other datasets.

You can install relevant dependencies for editing the public documentation via:
```sh
pip install -e .[docs]
```

It is recommended to uses [sphinx-autobuild](https://github.com/sphinx-doc/sphinx-autobuild) (installed above) to edit and view the documentation. You can run:
```sh
sphinx-autobuild docs docs/_build/html
```

## Adding new datasets
TODO


## Questions?

If you have any questions about contributing, please open an issue or reach out to the maintainers.
