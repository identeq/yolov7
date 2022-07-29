"""
-- Created by Pravesh Budhathoki
-- Treeleaf Technologies Pvt. Ltd.
-- Created on 2022-07-28 
"""
from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = []
try:
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
except IOError as e:
    print(e)

setup(
    name='yolov7',
    version='1.0',
    description="",
    long_description=readme + '\n',
    author="Pravesh Kaji Budhathoki",
    author_email='pravesh.budhathoki@treeleaf.ai',
    url='',
    packages=find_packages(),
    package_dir={},
    package_data={},
    install_requires=requirements,
    license="",
    zip_safe=False,
    keywords=''
)