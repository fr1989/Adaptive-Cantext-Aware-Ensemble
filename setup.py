from setuptools import setup, find_packages

setup(
    name="adaptive_ensemble",
    version="0.1.0",
    description="Adaptive Context-Aware Ensemble Learning package",
    author="F. Rabbani",
    author_email="frabbani89@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "joblib"
    ],
)
