---
orphan: true
---

You can install relevant dependencies for editing the public documentation via:
```sh
pip install -e .[docs]
```

It is recommended to uses [sphinx-autobuild](https://github.com/sphinx-doc/sphinx-autobuild) (installed above) to edit and view the documentation. You can run:

```sh
sphinx-autobuild docs docs/_build/html
```
