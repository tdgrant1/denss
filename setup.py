#!/usr/bin/env python

from distutils.core import setup

setup(name='denss',
      version='1.0.0',
      author='Thomas Grant',
      author_email='tgrant@hwi.buffalo.edu',
      py_modules=['saxstats'],
      scripts=['denss.py','superdenss','ali2xyz.py','best_enantiomers.sh'],
      url='https://github.com/tdgrant1/denss/',
      license='GPLv3',
      description='Calculate electron density from solution scattering data.',
      long_description=open('README.md').read(),
      requires=['numpy', 'scipy'],
     )
