from setuptools import setup, find_packages

setup(
    name='toxic_service',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'kafka-python',
        'python-dotenv',
        'tensorflow==2.11.0',
    ],
)