from setuptools import find_packages, setup

setup(
    name='ml_autotrader',
    packages=find_packages(),
    version='0.1.0',
    description='Machine learning for automated trading research',
    author='Your Name',
    license='MIT',
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',
        'xgboost',
        'matplotlib',
        'seaborn',
        'yfinance',
        'joblib',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)