from setuptools import setup, find_packages

setup(
    name="examples",
    description="",
    author="Andrew Hah",
    author_email="hahdawg@yahoo.com",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    package_data = {
        "": ["*.yaml", "*.ini"]
    },
    install_requires=[
        "numpy==1.19.2",
        "pandas==1.1.5",
        "PyYAML==5.4.1",
        "scikit_learn==0.24.1"
    ],
    zip_safe=False
)
