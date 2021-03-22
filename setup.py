from setuptools import setup

setup(name='pilates',
      version='0.0.1',
      description='Pilates Is a Library of Accounting Tools for EconomistS. Its purpose is to help data creation for Accounting, Finance, and Economic Research.',
      url='https://github.com/milonemario/pilates',
      author='Mario Milone',
      author_email='milonemario@gmail.com',
      license='BSD-3',
      packages=['pilates'],
      install_requires=[
            'numpy',
            'pandas',
            'scipy',
            'sklearn',
            'pyarrow',
            'pyyaml',
            'fredapi',
            'rapidfuzz',
            'psycogpg2',
            'wget',
            'gzip',
      ],
      zip_safe=False)
