import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name = 'macroecotools',
      version= '0.4.0',
      description = 'Tools for conducting macroecological analyses',
      author = "Xiao Xiao, Katherine M. Thibault, David J. Harris, Elita Baldridge, and Ethan P. White",
      url = 'https://github.com/weecology/macroecotools',
      packages = ['macroecotools', 'macroeco_distributions'],
      license = 'MIT',
      install_requires = ['numpy>=1.6', 'scipy>=0.12', 'pandas>=0.13']
)
