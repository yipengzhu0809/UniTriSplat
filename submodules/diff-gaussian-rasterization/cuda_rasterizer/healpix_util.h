#ifndef CUDA_RASTERIZER_HEALPIXUTIL_H_INCLUDED
#define CUDA_RASTERIZER_HEALPIXUTIL_H_INCLUDED

#include <cuda_runtime.h>
#include "auxiliary.h"
#include <stdio.h>
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#define CUDA_ASSERT(condition, message) if (!(condition)) { printf("Assert failed: %s\n", message); asm("trap;"); }
struct NestParams {
    int ix;
    int iy;
    int face_num;
    int nest_idx;
};

__constant__ int d_jrll[] = { 2,2,2,2,3,3,3,3,4,4,4,4 };
__constant__ int d_jpll[] = { 1,3,5,7,0,2,4,6,1,3,5,7 };

namespace healpix_util {
    class Healpix_Nested_CUDA{
        public:
            int order_;
            int nside_;
            int npix_;
            int npface_;
            double fact1_;
            double fact2_;
            double crpdr;
            double crmdr;
        public:
            __device__ Healpix_Nested_CUDA(): order_(0), nside_(1), npix_(12), npface_(1), fact1_(0.333333), fact2_(0.666666), crpdr(0), crmdr(0) {}
            __device__ Healpix_Nested_CUDA(int order, int nside, int npix, int npface, double fact1, double fact2)
                : order_(order), nside_(nside), npix_(npix), npface_(npface), fact1_(fact1), fact2_(fact2) {}

            __device__ void set(int order){
                order_  = order;
                nside_  = int(1) << order;
                npix_   = 12 * nside_ * nside_;
                npface_ = nside_ * nside_;
                fact2_  = 4.0 / npix_;
                fact1_  = (nside_ << 1) * fact2_;
                crpdr = 0;
                crmdr = 0;
            }    
    };



    template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

    struct HealpixState{
        Healpix_Nested_CUDA* healpix_gp;
        int2* d_stk_data;
        static HealpixState fromChunk(char*& chunk, int P, int stack_elements_per_thread, int omax); //HealpixState is a return type
    };

    template<typename T> 
	size_t required(int P, int stack_elements_per_thread, int omax)
	{
		char* size = nullptr;
		T::fromChunk(size, P, stack_elements_per_thread, omax);
		return ((size_t)size) + 128;
	}

    __device__ inline uint64_t d_spread_bits(uint64_t v) {

        uint64_t res = v & 0xffffffff;
        res = (res^(res<<16)) & 0x0000ffff0000ffff;
        res = (res^(res<< 8)) & 0x00ff00ff00ff00ff;
        res = (res^(res<< 4)) & 0x0f0f0f0f0f0f0f0f;
        res = (res^(res<< 2)) & 0x3333333333333333;
        res = (res^(res<< 1)) & 0x5555555555555555;

        return res;
    }

    __device__ inline int d_spread_bits(int v) { 

        int res = v & 0xffff; //Restrict the input to 16 bits, the & is AND operation
        res = (res ^ (res << 8)) & 0x00ff00ff;
        res = (res ^ (res << 4)) & 0x0f0f0f0f;
        res = (res ^ (res << 2)) & 0x33333333;
        res = (res ^ (res << 1)) & 0x55555555;

        return res;
    }

    __device__ inline int d_compress_bits(int v) {

        int res = v & 0x55555555;
        res = (res ^ (res >> 1)) & 0x33333333;
        res = (res ^ (res >> 2)) & 0x0f0f0f0f;
        res = (res ^ (res >> 4)) & 0x00ff00ff;
        res = (res ^ (res >> 8)) & 0x0000ffff;

        return res;
    }

    // Fast integer square root (for small values)
    __device__ inline int isqrt_cuda(int v) {
        return (int)sqrtf((float)v);
    }

    // Convert (ring_index, pixel_in_ring) to NESTED pixel index
    // Based on official HEALPix ring2nest implementation
    // ir: ring index (1 to 4*nside-1)
    // ip: pixel index within the ring (0 to ring_pixels-1)
    __device__ inline int ring2nest_cuda(int ir, int ip, int nside, int order) {
        int npix = 12 * nside * nside;
        int ncap = 2 * nside * (nside - 1);
        int nl2 = 2 * nside;
        
        // Step 1: Convert (ir, ip) to global RING pixel index
        int pix_ring;
        if (ir < nside) {
            // North polar cap: cumulative pixels = 2*ir*(ir-1), ring has 4*ir pixels
            pix_ring = 2 * ir * (ir - 1) + ip;
        } else if (ir <= 3 * nside) {
            // Equatorial belt
            pix_ring = ncap + (ir - nside) * 4 * nside + ip;
        } else {
            // South polar cap
            int ir_south = 4 * nside - ir;
            // Total pixels up to this ring from south = 2*ir_south*(ir_south+1)
            pix_ring = npix - 2 * ir_south * (ir_south + 1) + ip;
        }
        
        // Step 2: ring2xyf - convert RING pix to (ix, iy, face_num)
        int ix, iy, face_num;
        int iring, iphi, kshift, nr;
        
        if (pix_ring < ncap) {
            // North Polar cap
            iring = (1 + isqrt_cuda(1 + 2 * pix_ring)) >> 1;
            iphi = pix_ring + 1 - 2 * iring * (iring - 1);
            kshift = 0;
            nr = iring;
            face_num = (iphi - 1) / nr;
        } else if (pix_ring < npix - ncap) {
            // Equatorial region
            int ip_eq = pix_ring - ncap;
            int tmp = ip_eq >> (order + 2);  // ip_eq / (4*nside)
            iring = tmp + nside;
            iphi = ip_eq - tmp * 4 * nside + 1;
            kshift = (iring + nside) & 1;
            nr = nside;
            int ire = tmp + 1;
            int irm = nl2 + 1 - tmp;
            int ifm = iphi - (ire >> 1) + nside - 1;
            int ifp = iphi - (irm >> 1) + nside - 1;
            ifm >>= order;
            ifp >>= order;
            face_num = (ifp == ifm) ? (ifp | 4) : ((ifp < ifm) ? ifp : (ifm + 8));
        } else {
            // South Polar cap
            int ip_sp = npix - pix_ring;
            iring = (1 + isqrt_cuda(2 * ip_sp - 1)) >> 1;
            iphi = 4 * iring + 1 - (ip_sp - 2 * iring * (iring - 1));
            kshift = 0;
            nr = iring;
            iring = 2 * nl2 - iring;  // Convert to global ring index
            face_num = (iphi - 1) / nr + 8;
        }
        
        // Step 3: Compute (ix, iy) from ring info
        // jpll = {1,3,5,7,0,2,4,6,1,3,5,7}
        int jpll_val = d_jpll[face_num];
        int irt = iring - ((2 + (face_num >> 2)) * nside) + 1;
        int ipt = 2 * iphi - jpll_val * nr - kshift - 1;
        if (ipt >= nl2) ipt -= 8 * nside;
        
        ix = (ipt - irt) >> 1;
        iy = (-ipt - irt) >> 1;
        
        // Step 4: xyf2nest
        return (face_num << (2 * order)) + d_spread_bits(ix) + (d_spread_bits(iy) << 1);
    }

    __device__ inline int compute_order(int nside) {

        unsigned int res = 0;
        while (nside > 0x00FF) {res |= 8; nside >>= 8;}
        if (nside > 0x000F) {res |= 4; nside >>= 4;}
        if (nside > 0x0003) {res |= 2;nside >>= 2;}
        if (nside > 0x0001) {res |= 1;}
        return res;
    }

    __forceinline__ __device__ void max_pixradCUDA(Healpix_Nested_CUDA& healpix_gp, double& dr){
        double z_a = 2.0 / 3.0;
        double phi_a = M_PI / (4.0 * healpix_gp.nside_);
        double x_a = sqrt((1.0 - z_a) * (1.0 + z_a)) * cos(phi_a);
        double y_a = sqrt((1.0 - z_a) * (1.0 + z_a)) * sin(phi_a);
        double t1_a = 1.0 - 1.0 / healpix_gp.nside_;
        t1_a *= t1_a;

        double z_b = 1.0 - t1_a / 3.0;
        double x_b = sqrt((1.0 - z_b) * (1.0 + z_b));
        double y_b = 0.0;

        double cross_x = y_a * z_b - z_a * y_b;
        double cross_y = z_a * x_b - x_a * z_b;
        double cross_z = x_a * y_b - y_a * x_b;
        double cross_len = sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
        double dot = x_a * x_b + y_a * y_b + z_a * z_b;
        dr = atan2(cross_len, dot);  
    }

    __forceinline__ __device__ void nest2xyfCUDA(int pix, int &ix, int &iy, int &face_num, Healpix_Nested_CUDA healpix_gp){
        face_num = pix>>(2*healpix_gp.order_);
        pix &= (healpix_gp.npface_-1);
        ix = d_compress_bits(pix);
        iy = d_compress_bits(pix>>1);
    }

    __forceinline__ __device__ void pix2zphiCUDA(int pix, double &z, double &phi, Healpix_Nested_CUDA healpix_gp){
        bool have_sth;
        double sth;
        have_sth=false;
        int face_num, ix, iy;
        if (pix<0 || pix>=healpix_gp.npix_){
            printf("Error: pix out of range in pix2zphiCUDA: pix=%d, npix=%d\n", pix, healpix_gp.npix_);
        }
        nest2xyfCUDA(pix,ix,iy,face_num, healpix_gp);
        int jr = (d_jrll[face_num]<<healpix_gp.order_) - ix - iy - 1;
        int nr;
        if (jr<healpix_gp.nside_){
            nr = jr;
            double tmp = (nr * nr) * healpix_gp.fact2_;
            z = 1.0 - tmp;
            if (z>0.99) {sth = sqrt(tmp * (2.0 - tmp)); have_sth=true;}
        }
        else if (jr>3*healpix_gp.nside_){
            nr = healpix_gp.nside_ * 4 - jr;
            double tmp = (nr * nr) * healpix_gp.fact2_;
            z = tmp - 1.0;
            if (z<-0.99) {sth = sqrt(tmp * (2.0 - tmp)); have_sth=true;}
        }
        else{
            nr = healpix_gp.nside_;
            z = (2.0 * healpix_gp.nside_ - jr) * healpix_gp.fact1_;
        }
        int tmp = (d_jpll[face_num]) * nr + ix - iy;
        if (tmp >= 8 * nr) {
            printf("Error: tmp >= 8*nr, tmp=%d, nr=%d\n", tmp, nr);
            // return;
            tmp -= 8 * nr;
        }
        if (tmp < 0) tmp += 8 * nr;
        phi = (nr == healpix_gp.nside_) ? 0.75 * M_PI / 2 * tmp * healpix_gp.fact1_ : (0.5 * M_PI / 2 * tmp) / nr;
    }

    __forceinline__ __device__ void xyf2nestCUDA(uint x, uint y, uint face, int order, uint32_t& pix_id){
        uint64_t face_ = uint64_t(face);
        uint64_t x_ = uint64_t(x);
        uint64_t y_ = uint64_t(y);
        pix_id = ((face_)<<(2*order)) + d_spread_bits(x_) + (d_spread_bits(y_)<<1); 
    }

    __forceinline__ __device__ NestParams loc2_nest(const float z, const float phi, float sth, bool have_sth, int nside){
        NestParams result = {0, 0, 0, 0};
        float za = abs(z);
        float tt = phi * (2.0f / M_PIf);
        float four = 4.0f;
        fmoduloCUDA(tt, four); // in [0,4)
        float two_third = 2.0f / 3.0f;
        int order = compute_order(nside);
        if (za <= two_third){
            float temp1 = nside * (0.5f + tt);
            float temp2 = nside * (z * 0.75f);
            int jp = int(temp1 - temp2); // index of  ascending edge line
            int jm = int(temp1 + temp2); // index of descending edge line
            int ifp = jp >> order;  // in {0,4}
            int ifm = jm >> order;
            result.face_num = (ifp == ifm) ? (ifp | 4) : ((ifp < ifm) ? ifp : (ifm + 8));
            result.ix = jm & (nside - 1);
            result.iy = nside - (jp & (nside - 1)) - 1;
            result.nest_idx = (int(result.face_num) << (2 * order)) + d_spread_bits(result.ix) + (d_spread_bits(result.iy) << 1);
        }
        else{
            int ntt = min(3, int(tt));
            float tp = tt - ntt;
            float tmp = ((za<0.99)||(!have_sth)) ?
                    nside*sqrt(3*(1-za)) :
                    nside*sth/sqrt((1.+za)/3.);
            int jp = int(tp*tmp); // increasing edge line index
            int jm = int((1.0f - tp)*tmp); // decreasing edge line
            jp = min(jp, nside - 1); // for points too close to the boundary
            jm = min(jm, nside - 1);
            if (z >= 0){
                result.ix = nside - jm - 1;
                result.iy = nside - jp - 1;
                result.face_num = ntt;
                result.nest_idx = (int(result.face_num) << (2 * order)) + d_spread_bits(result.ix) + (d_spread_bits(result.iy) << 1);
            }
            else{
                result.ix = jp;
                result.iy = jm;
                result.face_num = ntt + 8;
                result.nest_idx = (int(result.face_num) << (2 * order)) + d_spread_bits(result.ix) + (d_spread_bits(result.iy) << 1);
            }
        }
        return result;
    }

    __forceinline__ __device__ void lonlat2_nest_xyf(float lon, float lat, int& ix, int& iy, int& face, int& nest_index, const int nside){
        float theta = M_PIf / 2.0 - lat; // Convert latitude to colatitude [0, π]
        float phi = lon + M_PIf;        // Convert longitude to [0, 2π]
        float z = cos(theta);
        NestParams result;
        const float epsilon = 1e-6;
        // CUDA_ASSERT(theta >= -epsilon && theta <= M_PIf + epsilon, "theta out of range");
        if (theta < 0.0f){
            theta = 0.0f + epsilon;
        }
        else if (theta > M_PIf){
            theta = M_PIf - epsilon;
        }
        if (theta < 0.01 || theta > M_PIf - 0.01) {
            // Near the poles, sth is not well-defined
            result = loc2_nest(z, phi, sin(theta), true, nside);
        } else {
            result = loc2_nest(z, phi, 0, false, nside);
        }
        ix = result.ix;
        iy = result.iy;
        face = result.face_num;
        nest_index = result.nest_idx;
    }


} // namespace healpix_util

__device__ void check_pixelCUDA(
    const int    o,
    const int    order,
    const int    omax,
    const int    zone,
    const int    pix,
    int2*        my_stk,
    const bool   inclusive,
    int&         stack_size,
    int&         stacktop,
    uint32_t&    tile_num,
    const int    s_size,
    int&         myoff,
    uint32_t     touch_num,
    int*         tilesKey_touched = nullptr
);

__global__ void query_disc_downsample(
    int P,
    float2* lonlat_image,
    float* radii,
    int fact,
    const int nside,
    healpix_util::Healpix_Nested_CUDA* healpix_gp,
    uint32_t* tiles_touched,
    int2* d_stk_data,
    int* tilesKey_touched = nullptr,
    uint32_t* point_offsets = nullptr
);

__global__ void query_hp_from_equi(
    int W, int H, int order,
    float* out_color,
	float* depth,
	float* hp_color,
	float* hp_depth
);

__global__ void query_equi_from_hp(
    int W, int H, int order,
    float* original_image,
	float* original_hp,
	float* original_depth = nullptr,
    float* depth_hp = nullptr
);

// Optimized version: skip healpix initialization (for second call)
__global__ void query_disc_downsample_skip_init(
    int P,
    float2* lonlat_image,
    float* radii,
    int fact,
    const int nside,
    healpix_util::Healpix_Nested_CUDA* healpix_gp,  // Already initialized
    uint32_t* tiles_touched,
    int2* d_stk_data,
    int* tilesKey_touched,
    uint32_t* point_offsets
);

// Rectangle bounding box method: directly enumerate pixels instead of quadtree
__global__ void query_disc_rect(
    int P,
    float2* lonlat_image,
    float* radii,
    int fact,
    const int nside,
    uint32_t* tiles_touched,
    int* tilesKey_touched,
    uint32_t* point_offsets
);


#endif // CUDA_RASTERIZER_HEALPIXUTIL_H_INCLUDED