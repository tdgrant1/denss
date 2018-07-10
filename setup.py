#!/usr/bin/env python

from distutils.core import setup

execfile('saxstats/_version.py')

setup(name='denss',
      version=__version__,
      author='Thomas Grant',
      author_email='tgrant@hwi.buffalo.edu',
      packages=['saxstats'],
      scripts=['bin/denss.py','bin/superdenss','bin/best_enantiomers.sh','bin/fsc2res.py','bin/sasrec.py','bin/rho2dat.py'],
      url='https://github.com/tdgrant1/denss/',
      license='GPLv3',
      description='Calculate electron density from solution scattering data.',
      long_description=open('README.md').read(),
      requires=['numpy', 'scipy'],
     )
