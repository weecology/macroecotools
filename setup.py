import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name = 'macroecotools',
      version= '0.4.0',
      description = 'Tools for conducting macroecological analyses',
      author = "Xiao Xiao, Kate Thibault, David Harris, Elita Baldridge, and Ethan White",
      url = 'https://github.com/weecology/macroecotools',
      packages = ['macroecotools', 'macroeco_distributions'],
      license = 'MIT',
      install_requires = ['numpy>=1.6', 'scipy>=0.12', 'pandas>=0.13'],
      classifiers=['Intended Audience :: Science/Research',
                   'License :: OSI Approved :: MIT License',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 2', ],

)
