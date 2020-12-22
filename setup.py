# @Author: charles
# @Date:   03-06-2019
# @Email:  charles@goldspot.ca
# @Filename: setup.py
# @Last modified by:   charles
# @Last modified time: 2020-12-21 22:12:18


import os
from setuptools import setup, find_packages

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements'
install_requires = []  # To populate by reading the requirements file
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(
    name='gIms',
    packages=find_packages(),
    description='Command line interface for Geospatial Image Segmentation',
    version='0.0.1',
    url='https://github.com/clberube/gims',
    author='clberube',
    author_email='charles.lafreniere-berube@polymtl.ca',
    keywords=['deep learning', 'GIS', 'unet', 'segmentation'],
    install_requires=install_requires,
    )
