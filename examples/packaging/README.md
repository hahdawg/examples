# Packaging Steps
1. Create source code distribution and wheel.
```console
python setup.py sdist bdist_wheel
```
2. Install Package
```console
pip install -e <<my_package>>
```
