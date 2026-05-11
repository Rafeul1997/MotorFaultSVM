from setuptools import setup, find_packages

setup(
    name="motorfaultsvm",
    version="1.0.0",
    author="Abdul Rafeul Mallick",
    description="Motor Fault Detection using Support Vector Machine",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn"
    ],
)
