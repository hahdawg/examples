# Packaging Steps
1. Create distribution packages
```console
python3 setup.py sdist bdist_wheel
```
2. Install Package
```console
pip install -e <<mypackage>>
```
