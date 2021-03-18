from setuptools import setup

setup(name='pilates',
      version='0.0.1',
      description='Python library to create data for Accounting, Finance, and Economic Research.',
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
            'pandas_market_calendars',
            'fredapi',
            'rapidfuzz',
      ],
      zip_safe=False)
