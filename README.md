**This is the Thermally Induced Deformation Analysis Library (TIDAL).**

Copyright (C) 2021 Mechanics and Energy Laboratory, Northwestern University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

--------------------------------------------------------------------------------

TIDAL is a Python package that implements the most advanced and up-to-date tools
for the analysis of the thermally induced deformation of soils. It was developed
by the Mechanics and Energy Laboratory at Northwestern University, directed by
Prof. Alessandro F. Rotta Loria https://sites.northwestern.edu/rottaloria/ .


The primary author of the code is Dr. Jibril B. Coulibaly who can be contacted
at jibril.coulibaly@gmail.com. TIDAL is developed for the scientific community.
The code is written mainly for robustness, readability and tracktability to the
original literature rather than performance. Given the low computational cost of
the involved calculations, potential optimization is wilfully avoided when it
could potentially impair understanding and readability.

**Package structure, files and directories**

* README: this file
* LICENSE: the GNU General Public License (GPL) version 3
* core: sub-package, thermally induced deformation analysis modules
* data: sub-package, useful tabulated raw data and results of the literature
* example: test cases presented in the paper by Coulibaly and Rotta Loria, 2022
* template: (To be written, to simplify analysis for new users of TIDAL)

Documentation is available through docstrings provided with the code.  
The top-level TIDAL directory must be accessible to the PYTHONPATH.

**When using TIDAL please cite**:

Coulibaly, J. B. and Rotta Loria, A. F., 2022. Thermally induced deformation of
soils: a Critical Revision of experimental methods. _Journal J_ Vol, Page.
DOI: https://doi.org/XXXXXXXXXXXXXX
