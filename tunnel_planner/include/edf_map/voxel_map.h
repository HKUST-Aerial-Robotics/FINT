#ifndef _VOXEL_MAP_H
#define _VOXEL_MAP_H

#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <iostream>

using namespace std;
using namespace Eigen;

template <class voxel_type>
class voxel_map{

public:

    voxel_map(double MAP_RES, Matrix<double,3,2>& MAP_LIM): map_data(ceil(abs((MAP_LIM(0,1) - MAP_LIM(0,0))/MAP_RES)) * ceil(abs((MAP_LIM(1,1) - MAP_LIM(1,0))/MAP_RES)) * ceil(abs((MAP_LIM(2,1) - MAP_LIM(2,0))/MAP_RES))){
        map_res = MAP_RES;
        inv_map_res = 1.0 / map_res;

        map_size.x() = ceil(abs((MAP_LIM(0,1) - MAP_LIM(0,0))/MAP_RES));
        map_size.y() = ceil(abs((MAP_LIM(1,1) - MAP_LIM(1,0))/MAP_RES));
        map_size.z() = ceil(abs((MAP_LIM(2,1) - MAP_LIM(2,0))/MAP_RES));

        xmin = MAP_LIM(0,0);
        ymin = MAP_LIM(1,0);
        zmin = MAP_LIM(2,0);

        xmax = MAP_LIM(0,0) + MAP_RES * map_size.x();
        ymax = MAP_LIM(1,0) + MAP_RES * map_size.y();
        zmax = MAP_LIM(2,0) + MAP_RES * map_size.z();

        // map_data.resize(map_size(0) * map_size(1) * map_size(2));
    }

    inline bool in_map(Vector3d pos){
        return pos.x() >= xmin && pos.y() >= ymin && pos.z() >= zmin && 
                pos.x() <= xmax && pos.y() <= ymax && pos.z() <= zmax;
    }

    inline bool in_map(Vector3i coord){
        return coord.x() >= 0 && coord.y() >= 0 && coord.z() >= 0 && 
                coord.x() < map_size.x() && coord.y() < map_size.y() && coord.z() < map_size.z();
    }

    inline bool in_map(int x, int y, int z){
        return x >= 0 && y >= 0 && z >= 0 && 
                x < map_size.x() && y < map_size.y() && z < map_size.z();
    }

    inline Vector3i pos2coord(Vector3d pos){
        return Vector3i(static_cast<int>(floor((pos.x()-xmin)/map_res)), 
                        static_cast<int>(floor((pos.y()-ymin)/map_res)),
                        static_cast<int>(floor((pos.z()-zmin)/map_res)));
    }

    inline Vector3i pos2coord(double x, double y, double z){
        return Vector3i(static_cast<int>(floor((x-xmin)/map_res)), 
                        static_cast<int>(floor((y-ymin)/map_res)),
                        static_cast<int>(floor((z-zmin)/map_res)));
    }

    inline unsigned int pos2idx(Vector3d pos){
        return coord2idx(Vector3i(static_cast<int>(floor((pos.x()-xmin)/map_res)), 
                        static_cast<int>(floor((pos.y()-ymin)/map_res)),
                        static_cast<int>(floor((pos.z()-zmin)/map_res))));
    }

    inline unsigned int coord2idx(int x, int y, int z){
        return x*map_size.y()*map_size.z() + y * map_size.z() + z;
    }
    
    inline unsigned int coord2idx(Vector3i idx){
        return idx.x()*map_size.y()*map_size.z() + idx.y() * map_size.z() + idx.z();
    }

    inline Vector3i idx2coord(unsigned int array_idx){
        unsigned int z = array_idx % (map_size.z());
        unsigned int y = ((array_idx - z) / map_size.z()) % map_size.y();
        unsigned int x = (array_idx - z - y * map_size.z()) / (map_size.y()*map_size.z());
        return Vector3i(x,y,z);
    }

    inline Vector3d idx2pos(unsigned int array_idx){
        int z = array_idx % (map_size.z());
        int y = ((array_idx - z) / map_size.z()) % map_size.y();
        int x = (array_idx - z - y * map_size.z()) / (map_size.y()*map_size.z());
        return (Vector3i(x,y,z).cast<double>() + Vector3d(0.5, 0.5, 0.5)) * map_res + Vector3d(xmin,ymin,zmin);
    }

    inline Vector3d coord2pos(Vector3i coord){
        return (coord.cast<double>() + Vector3d(0.5, 0.5, 0.5)) * map_res + Vector3d(xmin,ymin,zmin);
    }

    Vector3d clamp_point_at_boundary(const Vector3d& pt, const Vector3d& origin_pt){
    
        Vector3d diff = pt - origin_pt;
        Vector3d max_tc = Vector3d(xmax,ymax,zmax) - origin_pt;
        Vector3d min_tc = Vector3d(xmin,ymin,zmin) - origin_pt;
        double min_t = Vector3d(xmax-xmin+map_res,ymax-ymin+map_res,zmax-zmin+map_res).norm();

        for(unsigned int i = 0; i < 3; i++){
            if(fabs(diff(i)) > 0){
                double t1 = max_tc(i) / diff(i);
                if(t1 > 0 && t1 < min_t) min_t = t1;

                double t2 = min_tc(i) / diff(i);
                if(t2 > 0 && t2 < min_t) min_t = t2;
            }
        }

        return origin_pt + (min_t - map_res * 0.1) * diff;
    }


    Vector3i map_size;
    double map_res;
    double inv_map_res;
    vector<voxel_type> map_data;
    double xmin, ymin, zmin;
    double xmax, ymax, zmax;

private:

};

#endif