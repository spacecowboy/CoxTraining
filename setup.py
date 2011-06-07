#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy

numpy_include = numpy.get_include()

cox_ext = Extension('cox_error_in_c',
          sources = ['src/C_ext/cox_error_in_c.c'],
          include_dirs = [numpy_include],
          extra_compile_args = ['-std=c99'])

setup(name = 'CoxTraining',
      version = '0.1',
      description = 'An error function for my aNeuralN package.',
      author = 'Jonas Kalderstam',
      author_email = 'jonas@kalderstam.se',
      url = 'https://github.com/spacecowboy/CoxTraining',
      packages = ['survival', 'survival.tests'],
      package_dir = {'': 'src'},
      package_data = {},
      ext_package = 'survival',
      ext_modules = [cox_ext],
      requires = ['numpy', 'matplotlib'],
     )
