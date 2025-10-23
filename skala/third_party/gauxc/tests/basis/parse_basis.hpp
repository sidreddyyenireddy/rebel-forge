/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */
#include <gauxc/basisset.hpp>
#include <gauxc/molecule.hpp>
#include <string>

namespace GauXC {

BasisSet<double> parse_basis( const Molecule& mol,
                              std::string     fname,
                              SphericalType   sph    );

}
