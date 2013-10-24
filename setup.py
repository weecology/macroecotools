import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name = 'macroecotools',
      version= '0.1',
      description = 'Tools for conducting macroecological analyses',
      author = "Xiao Xiao, Ethan White and Katherine Thibault",
      url = 'https://github.com/weecology/macroecotools',
      packages = ['macroecotools', 'macroeco_distributions'],
      license = 'MIT',
      long_description = read('README.md'),
      install_requires = ['numpy>=1.6', 'scipy>=0.12']
)
