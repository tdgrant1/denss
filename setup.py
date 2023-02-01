#!/usr/bin/env python

from setuptools import setup

exec(compile(open('saxstats/_version.py').read(), 'saxstats/_version.py', 'exec'))

setup(name='denss',
      version=__version__,
      author='Thomas Grant',
      author_email='tdgrant@buffalo.edu',
      packages=['saxstats'],
      scripts=[
      'bin/denss.py',
      'bin/denss.align.py',
      'bin/denss.align2xyz.py',
      'bin/denss.align_by_principal_axes.py',
      'bin/denss.average.py',
      'bin/denss.align_and_average.py',
      'bin/denss.all.py',
      'bin/denss.refine.py',
      'bin/denss.fit_data.py',
      'bin/denss.calcfsc.py',
      'bin/denss.fsc2res.py',
      'bin/denss.rho2dat.py',
      'bin/denss.pdb2mrc.py',
      'bin/denss.get_info.py',
      'bin/denss.mrcops.py',
      'bin/denss.generate_reference.py',
      'bin/denss.select_enantiomers.py',
      'bin/denss.select_enantiomer.py',
      'bin/superdenss','bin/best_enantiomers.sh','bin/fsc2res.py'],
      url='https://github.com/tdgrant1/denss/',
      license='GPLv3',
      description='Calculate electron density from solution scattering data.',
      long_description=open('README.md').read(),
      requires=['numpy', 'scipy'],
     )
