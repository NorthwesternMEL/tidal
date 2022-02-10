"""
TIDAL : Thermally Induced Deformation Analysis Library
======================================================

Python 2
Scipy version 1.2.1
Numpy version 1.16.2
Jibril B. Coulibaly, jibril.coulibaly at gmail.com

Core sub-package:
  1. Module `thexp` : thermal expansion coefficient of substances
  2. Module `inteq` : integration of the differential and conservation equations
  3. Module `uq` : uncertainty quantification

Copyright (C) 2021 Mechanics and Energy Laboratory, Northwestern University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

See the README file in the top-level TIDAL directory.
"""

from thexp import *
from inteq import *
from uq import *

__all__ = ['thexp', 'inteq', 'uq']
__all__ += thexp.__all__
__all__ += inteq.__all__
__all__ += uq.__all__