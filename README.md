# Pilates

*P*ilates *I*s a *L*ibrary of *A*ccounting *T*ools for *E*conomist*S*.
Its purpose is to help data creation for Accounting, Finance, and Economic Research.

It is able to process financial and accounting data from
  - WRDS (CRSP, Compustat, IBES, ...)
  - Other useful sources (FRED, ...)

The library is flexible and uses separate modules (plugins) to process each data
sources, such as Compustat, CRSP, IBES, FRED, etc.

The library has the ability to directly download data from WRDS servers if necessary.

## Installation

To install, just run
python setup.py install

Given that the library is very new, users are encouraged to help develop it, fix
bugs and suggest new directions.

To install in develop mode (editable), run
python setup.py develop

Installing the library in develop mode allows the user to modify the source code
and have the changes applying immediately (after re-importing the library).
