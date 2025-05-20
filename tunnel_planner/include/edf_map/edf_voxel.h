/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/

#ifndef _EDF_VOXEL_H
#define _EDF_VOXEL_H

#include <atomic>

class edf_voxel{

public:
    enum edf_voxel_type{FREE, OCC, UNKNOWN};

    edf_voxel():type{UNKNOWN}, edf_value{0.0}, occ_value{0}{};

    std::atomic<int> type;
    double edf_value;
    std::atomic<int> occ_value;
};

#endif