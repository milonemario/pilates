from setuptools import find_packages, setup

setup(name='pilates',
      version='0.0.1',
      description=(
          """Pilates Is a Library of Accounting Tools for EconomistS.
          Its purpose is to help data creation for
          Accounting, Finance, and Economic Research."""),
      url='https://github.com/milonemario/pilates',
      author='Mario Milone',
      author_email='milonemario@gmail.com',
      license='BSD-3',
      # packages=['pilates', 'pilates.modules.audit'],
      packages=['pilates'] + ['pilates.modules.' + p for p in find_packages('./pilates/modules/')],
      package_data={'pilates': ['modules/**']},
      install_requires=[
          'numpy',
          'pandas',
          'numba',
          'scipy',
          'sklearn',
          'pyarrow',
          'pyyaml',
          'fredapi',
          'rapidfuzz',
          'psycopg2',
          'wget',
          'xlrd',
          'openpyxl',
          'country_converter'
      ],
      zip_safe=False)
