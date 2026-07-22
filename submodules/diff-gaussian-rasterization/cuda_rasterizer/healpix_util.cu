#include "healpix_util.h"

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
    int*         tilesKey_touched,
    int&         debug)
{
    if (zone==0) return;
    if (o<order){
        if (zone >= 3){
            int sdist = 2 * (order - o);
            int start_pix = pix << sdist;
            int end_pix = (pix + 1) << sdist;
            // printf("Add range [%d, %d) from order %d pixel %d (zone %d)\n", start_pix, end_pix, o, pix, zone);
            tile_num += end_pix - start_pix;
            if (tilesKey_touched != nullptr){
                for (int tp = start_pix; tp < end_pix; tp++){
                    tilesKey_touched[myoff] = tp;
                    myoff += 1;
                    // if (myoff > touch_num)
                    // {
                    //     printf("Error: myoff exceeded touch_num in check_pixelCUDA for pixel %d, order %d\n", pix, o);
                    //     debug = 1;
                    // }
                    
                }
            }
        }else{
            for (int i=0 ; i<4; ++i){
                int subpix = 4 * pix + 3 - i;
                my_stk[stack_size] = {subpix, o + 1};
                stack_size++;
                // if (stack_size < 0 || stack_size >= s_size) {
                //     printf("1Error: stack_size exceeded s_size in check_pixelCUDA for pixel %d, order %d\n", pix, o);
                // }
            }
        }
    }
    else if (o > order){
        if (zone >= 2){
            int parent_pix = pix >> (2 * (o - order));
            // printf("Add single pixel %d (parent of order %d pixel %d, zone %d)\n", parent_pix, o, pix, zone);
            tile_num += 1;
            if (tilesKey_touched != nullptr){
                tilesKey_touched[myoff] = parent_pix;
                myoff += 1;
                // if (myoff > touch_num)
                // {
                //     printf("Error: myoff exceeded touch_num in check_pixelCUDA for pixel %d, order %d\n", pix, o);
                //     debug = 1;
                // }
            }
            stack_size = stacktop;
            // if (stack_size < 0 || stack_size >= s_size) {
            //     printf("Error: stack_size out of bounds after unwind in check_pixelCUDA for pixel %d, order %d: %d\n", pix, o, stack_size);
            // }
        }else{
            if (o < omax){
                for (int i=0; i < 4; ++i){
                    int subpix = 4 * pix + 3 - i;
                    my_stk[stack_size] = make_int2(subpix, o + 1);
                    stack_size++;
                    // if (stack_size < 0 || stack_size >= s_size) {
                    //     printf("2Error: stack_size exceeded s_size in check_pixelCUDA for pixel %d, order %d\n", pix, o);
                    // }
                }
            } else{
                int parent_pix = pix >> (2 * (o - order));
                // printf("Add single pixel %d (parent of order %d pixel %d, resolution limit, zone %d)\n", parent_pix, o, pix, zone);
                tile_num += 1;
                if (tilesKey_touched != nullptr){
                    tilesKey_touched[myoff] = parent_pix;
                    myoff += 1;
                    // if (myoff > touch_num)
                    // {
                    //     printf("Error: myoff exceeded touch_num in check_pixelCUDA for pixel %d, order %d\n", pix, o);
                    //     debug = 1;
                    // }
                }
                stack_size = stacktop;
                // if (stack_size < 0 || stack_size >= s_size) {
                //     printf("3Error: stack_size out of bounds after unwind in check_pixelCUDA for pixel %d, order %d: %d\n", pix, o, stack_size);
                // }
            }
        }
    } else{
        if (zone >=2){
            // printf("Add single pixel %d (zone %d)\n", pix, zone);
            tile_num += 1;
            if (tilesKey_touched != nullptr){
                tilesKey_touched[myoff] = pix;
                myoff += 1;
                // if (myoff > touch_num)
                // {
                //     printf("Error: myoff exceeded touch_num in check_pixelCUDA for pixel %d, order %d\n", pix, o);
                //     debug = 1;
                // }
            }
        } else if (inclusive){
            if (order < omax){
                stacktop = stack_size;
                for (int i = 0; i < 4; ++i){
                    int subpix = 4 * pix + 3 - i;
                    my_stk[stack_size] = make_int2(subpix, o + 1);
                    stack_size++;
                    // if (stack_size < 0 || stack_size >= s_size) {
                    //     printf("4Error: stack_size exceeded s_size in check_pixelCUDA for pixel %d, order %d\n", pix, o);
                    // }
                }
            }
            else{
            // printf("Add single pixel %d (resolution limit, zone %d)\n", pix, zone);
                tile_num += 1;
                if (tilesKey_touched != nullptr){
                    tilesKey_touched[myoff] = pix;
                    myoff += 1;
                    // if (myoff > touch_num)
                    // {
                    //     printf("Error: myoff exceeded touch_num in check_pixelCUDA for pixel %d, order %d\n", pix, o);
                    //     debug = 1;
                    // }
                }
            }
        }
    }
        
}

__global__ void query_disc_downsample(
    int P,
    float2* lonlat_image,
    float* radii,
    int fact,
    const int nside,
    healpix_util::Healpix_Nested_CUDA* healpix_gp,
    uint32_t* tiles_touched,
    int2* d_stk_data,
    int* tilesKey_touched,
    uint32_t* point_offsets)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P) return;

    const float MIN_RADIUS = 1e-7f;
    double r = double(radii[idx]);
    if (!isfinite(r) || r <= double(MIN_RADIUS)){
        tiles_touched[idx] = 0u;
        return;
    }
    
    uint32_t touch_num = tiles_touched[idx];
    double radius;
    bool inclusive = (fact!=0);
    const float epsilon = 1e-6;
    float lon = lonlat_image[idx].x;
    float lat = lonlat_image[idx].y;

	float theta = M_PI / 2.0f - lat;
	float phi0 = lon + M_PI;
	uint32_t total_pixels = 12U * (uint32_t)nside * (uint32_t)nside;
    uint32_t off = 0;
    if (point_offsets != nullptr){
        off = (idx == 0) ? 0 : point_offsets[idx - 1];
    }
    int* myKey_touched = tilesKey_touched ? &tilesKey_touched[off] : nullptr;
	
	// if (theta < 0.0f || theta > M_PI || phi0 < 0.0f || phi0 > 2.0f * M_PI) {
	// 	printf("Error: lonlat_image out of bounds for Gaussian %d: (%f, %f)\n", idx, lon, lat);
	// }
	float2 ptg = {theta, phi0};
	normalize_pt(ptg);
	// if (ptg.x < 0.0f || ptg.x > M_PI || ptg.y < 0.0f || ptg.y > 2.0f * M_PI) {
	// 	printf("Error: normalized pt out of bounds for Gaussian %d: (%f, %f)\n", idx, ptg.x, ptg.y);
	// }
	radius = radii[idx];
	// if (radius >= M_PI){
	// 	tiles_touched[idx] = total_pixels;
	// 	return;
	// }
	int order = round(log2(nside));
	int oplus = 0;
	if (inclusive){
		oplus = round(log2(fact));
	} else {
		oplus = 0;
	}
	int omax = order + oplus;
	int base = idx * (omax + 1);
	int s_size = 12 + 6 * omax;
	float cosrad = cos(radius);

	for (int o = idx * (omax + 1); o < (idx + 1) * (omax + 1); o++) {
		int current_order = o % (omax + 1);
		healpix_gp[o].set(current_order);
		double dr;
		healpix_util::max_pixradCUDA(healpix_gp[o],dr);
		healpix_gp[o].crpdr = (radius+dr>M_PI) ? -1. : cos(radius+dr);
		healpix_gp[o].crmdr = (radius-dr<0.) ?  1. : cos(radius-dr);
	}
	int2* my_stk = &d_stk_data[idx * s_size];
	int stack_size = 0;
	uint32_t tile_num = 0;
	for (int i = 0; i < 12; ++i) {
        my_stk[stack_size] = {11 - i, 0};  // .x = pix, .y = o
        stack_size++;
    }
	int stacktop = 0;
    int myoff = 0;
	while(stack_size > 0){
        int debug = 0;
		stack_size--;
		if (stack_size < 0 || stack_size >= s_size) {printf("Error: stack_size out of bounds for Gaussian %d: %d\n", idx, stack_size);}
        int2 current = my_stk[stack_size];
		int pix = current.x;
        int o_n = current.y;
		int o = idx * (omax + 1) + o_n;
		double z,phi;
		healpix_util::pix2zphiCUDA(pix, z, phi, healpix_gp[o]);
		float3 vptg = {sin(ptg.x)*cos(ptg.y), sin(ptg.x)*sin(ptg.y), cos(ptg.x)};
		double cangdist = vptg.z * z + cos(ptg.y - phi) * sqrt((1. - vptg.z * vptg.z) * (1. - z * z));
		if (cangdist > healpix_gp[o].crpdr){
			int zone = (cangdist < cosrad) ? 1 : ((cangdist <= healpix_gp[o].crmdr) ? 2 : 3);
			// if (idx == 10){
			check_pixelCUDA(o_n, order, omax, zone, pix, my_stk, inclusive, stack_size, stacktop, tile_num, s_size, myoff, touch_num, myKey_touched, debug);
			// }
            // if (debug != 0 && myKey_touched != nullptr && idx <= 10){
            //     printf("Debug info for Gaussian %d: radius %f, theta %f, phi %f, order %d, pix %d, zone %d, stack_size %d, myoff %d, touch_num %d\n", idx, radius, theta, phi0, o_n, pix, zone, stack_size, myoff, touch_num);
            // }
		}
	}
	uint32_t old_num = tiles_touched[idx];
    if ((point_offsets != nullptr) && (int(old_num) != int(tile_num))){
        printf("Warning: tiles_touched changed after downsampling for Gaussian %d: old %d, new %d, radius %f\n", idx, old_num, tile_num, radius);
    }

    tiles_touched[idx] = tile_num;
    // if(tile_num > 1){
    //     tiles_touched[idx] = tile_num;
    // }
    // else{
    //     tiles_touched[idx] = 0;
    // }
	
	// if (idx == 10){
	// 	printf("radius of Gaussian %d: %f, tiles_touched after downsampling: %d, old_tiles_touched: %d, theata: %f, phi: %f, key touched: ", idx, radius, tiles_touched[idx], old_num, theta, phi0);
    //     // printf("order: %d, nside: %d, total_pixels: %d\n", order, nside, total_pixels);
    //     if (myKey_touched == nullptr) return;
    //     for (uint32_t t = 0; t < tile_num; t++){
    //         printf("%d ", myKey_touched[t]);
    //     }
    //     printf("\n");
	// }
} 


__global__ void query_hp_from_equi(
    int W, int H, int order,
    float* out_color,
	float* depth,
	float* hp_color,
	float* hp_depth
)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= W * H) return;

    int N_side = 1 << order;
    int pix_equi_id = idx;
    int x = pix_equi_id % W;
    int y = pix_equi_id / W;
    float2 lonlat;
    // equipixel2Lonlat(x, y, W, H, lonlat);
    equipixel2Lonlat_center(x, y, W, H, lonlat);
    int ix, iy, face, nest_id;
    if(pix_equi_id < 0 || pix_equi_id >= W * H) {
        printf("Error: equi pixel id out of bounds: %d\n", pix_equi_id);
        return;
    }
    healpix_util::lonlat2_nest_xyf(lonlat.x, lonlat.y, ix, iy, face, nest_id, N_side);
    if(nest_id < 0 || nest_id >= 12 * N_side * N_side) {
        printf("Error: nest id out of bounds: %d\n", nest_id);
        return;
    }

    for (int ch = 0; ch < 3; ch++){
        out_color[ch*(W*H) + pix_equi_id] = hp_color[ch*(12*N_side*N_side) + nest_id];
    }
    depth[pix_equi_id] = hp_depth[nest_id];
}

__global__ void query_equi_from_hp(
    int W, int H, int order,
    float* original_image,
	float* original_hp,
	float* original_depth,
    float* depth_hp
){
    auto idx = cg::this_grid().thread_rank();
    
    int N_side = 1 << order;
    if (idx >= N_side * N_side * 12) return;
    int pix_hp_id = idx;
    double z, phi;
    healpix_util::Healpix_Nested_CUDA hp;
    hp.set(order);
    healpix_util::pix2zphiCUDA(pix_hp_id, z, phi, hp);

    if (phi < 0){
		phi = 0.00001f;
	}
	if (phi > 2.0 * M_PIf){
		phi = 2.0f * M_PIf - 0.00001f;
	}
	double theta = acos(z);
	float pix_lat = M_PIf / 2.0f - theta;
	float pix_lon = phi - M_PIf;

    int2 pix_equi_id;
    lonlat2Equipixel_index(pix_lon, pix_lat, W, H, pix_equi_id);
    // lonlat2Equipixel(pix_lon, pix_lat, W, H, pix_equi_id);

    for (int ch = 0; ch < 3; ch++){
        original_hp[ch*(12*N_side*N_side) + pix_hp_id] = original_image[int(ch*(W*H) + round(pix_equi_id.x) + round(pix_equi_id.y)*W)];
        if(original_depth != nullptr){
            depth_hp[pix_hp_id] = original_depth[int(round(pix_equi_id.x) + round(pix_equi_id.y)*W)];
        }
    }
}

// ============ Rectangle bounding box method ============
// Instead of quadtree traversal, directly enumerate pixels in theta/phi bounds
__global__ void query_disc_rect(
    int P,
    float2* lonlat_image,
    float* radii,
    int fact,
    const int nside,
    uint32_t* tiles_touched,
    int* tilesKey_touched,
    uint32_t* point_offsets)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P) return;

    const float MIN_RADIUS = 1e-7f;
    float r = radii[idx];
    if (!isfinite(r) || r <= MIN_RADIUS){
        tiles_touched[idx] = 0u;
        return;
    }
    
    float lon = lonlat_image[idx].x;  // [-pi, pi]
    float lat = lonlat_image[idx].y;  // [-pi/2, pi/2]
    float theta_center = M_PIf / 2.0f - lat;  // [0, pi]
    float phi_center = lon + M_PIf;           // [0, 2*pi]
    float radius = r;
    
    // Use __sincosf for efficiency
    float sin_theta, cos_theta;
    __sincosf(theta_center, &sin_theta, &cos_theta);
    
    float cosrad = cosf(radius);
    float sin_theta_sq = sin_theta * sin_theta;
    
    // Compute z bounds (z = cos(theta))
    float theta_min = fmaxf(0.0f, theta_center - radius);
    float theta_max = fminf(M_PIf, theta_center + radius);
    float z_max = cosf(theta_min);  // top of disc
    float z_min = cosf(theta_max);  // bottom of disc
    
    // Get output offset for writing tile keys
    uint32_t off = 0;
    if (point_offsets != nullptr){
        off = (idx == 0) ? 0 : point_offsets[idx - 1];
    }
    int* myKey_touched = tilesKey_touched ? &tilesKey_touched[off] : nullptr;
    
    uint32_t tile_num = 0;
    
    // Precompute constants
    const float two_third = 2.0f / 3.0f;
    const float inv_3nside2 = 1.0f / (3.0f * nside * nside);
    const float two_over_3nside = 2.0f / (3.0f * nside);
    int order = 31 - __clz(nside);  // Fast log2(nside)
    
    // Compute ring range from z bounds
    int ir_min, ir_max;
    if (z_max > two_third) {
        ir_min = max(1, (int)ceilf(nside * sqrtf(3.0f * (1.0f - z_max))));
    } else if (z_max < -two_third) {
        ir_min = 4 * nside - (int)floorf(nside * sqrtf(3.0f * (1.0f + z_max)));
    } else {
        ir_min = (int)ceilf(nside * (2.0f - 1.5f * z_max));
    }
    
    if (z_min > two_third) {
        ir_max = max(1, (int)floorf(nside * sqrtf(3.0f * (1.0f - z_min))));
    } else if (z_min < -two_third) {
        ir_max = min(4 * nside - 1, 4 * nside - (int)ceilf(nside * sqrtf(3.0f * (1.0f + z_min))));
    } else {
        ir_max = (int)floorf(nside * (2.0f - 1.5f * z_min));
    }
    
    ir_min = max(1, ir_min);
    ir_max = min(4 * nside - 1, ir_max);
    
    // Iterate through rings
    for (int ir = ir_min; ir <= ir_max; ir++) {
        float z_ring, phi_start;
        int ring_pixels;
        
        if (ir < nside) {
            // North polar cap
            ring_pixels = 4 * ir;
            z_ring = 1.0f - (float)(ir * ir) * inv_3nside2;
            phi_start = M_PIf / (4.0f * ir);
        } else if (ir <= 3 * nside) {
            // Equatorial belt
            ring_pixels = 4 * nside;
            z_ring = (2.0f * nside - ir) * two_over_3nside;
            phi_start = ((ir - nside) & 1) ? 0.0f : (M_PIf / (4.0f * nside));
        } else {
            // South polar cap
            int ir_south = 4 * nside - ir;
            ring_pixels = 4 * ir_south;
            z_ring = -1.0f + (float)(ir_south * ir_south) * inv_3nside2;
            phi_start = M_PIf / (4.0f * ir_south);
        }
        
        // === Analytically compute phi half-range on this ring ===
        float one_minus_zring_sq = 1.0f - z_ring * z_ring;
        float sqrt_1_minus_z2 = sqrtf(fmaxf(0.0f, one_minus_zring_sq));
        float denom = sin_theta * sqrt_1_minus_z2;
        
        float dphi_half;  // Half-width of phi range on this ring
        
        if (denom < 1e-6f) {
            // Degenerate case: center at pole or ring at pole
            float cangdist_center = cos_theta * z_ring;
            if (cangdist_center >= cosrad) {
                dphi_half = M_PIf;  // Entire ring might be inside
            } else {
                continue;  // Ring doesn't intersect disc
            }
        } else {
            float cos_dphi_boundary = (cosrad - cos_theta * z_ring) / denom;
            
            if (cos_dphi_boundary >= 1.0f) {
                // Entire ring is outside the disc
                continue;
            } else if (cos_dphi_boundary <= -1.0f) {
                // Entire ring is inside the disc
                dphi_half = M_PIf;
            } else {
                // Partial intersection - add small margin for safety
                dphi_half = acosf(cos_dphi_boundary) + 0.01f;
            }
        }
        
        // Compute pixel range from phi range
        float dphi = 2.0f * M_PIf / ring_pixels;
        float inv_dphi = ring_pixels / (2.0f * M_PIf);
        
        // Precompute for exact check
        float sqrt_term = sqrtf(sin_theta_sq * one_minus_zring_sq);
        float vz_zring = cos_theta * z_ring;
        
        // Determine pixel range to check
        int ip_min, ip_max;
        bool full_ring = (dphi_half >= M_PIf - 1e-5f);
        
        if (full_ring) {
            ip_min = 0;
            ip_max = ring_pixels - 1;
        } else {
            // Convert phi bounds to pixel indices with margin
            float phi_lo = phi_center - dphi_half;
            float phi_hi = phi_center + dphi_half;
            
            // Compute candidate pixel range
            int ip_lo = (int)floorf((phi_lo - phi_start) * inv_dphi);
            int ip_hi = (int)ceilf((phi_hi - phi_start) * inv_dphi);
            
            // Check if we need to handle wraparound
            if (ip_lo < 0 || ip_hi >= ring_pixels) {
                // Wraparound case - check all pixels (safe fallback)
                ip_min = 0;
                ip_max = ring_pixels - 1;
            } else {
                ip_min = ip_lo;
                ip_max = ip_hi;
            }
        }
        
        // Clamp ip range to valid bounds to prevent illegal memory access
        ip_min = max(0, ip_min);
        ip_max = min(ring_pixels - 1, ip_max);
        
        // Iterate through pixels in this ring within phi range
        for (int ip = ip_min; ip <= ip_max; ip++) {
            float phi_pix = phi_start + ip * dphi;
            // Normalize phi_pix to [0, 2*pi)
            if (phi_pix >= 2.0f * M_PIf) phi_pix -= 2.0f * M_PIf;
            if (phi_pix < 0.0f) phi_pix += 2.0f * M_PIf;
            
            // Exact check: is this pixel inside the disc?
            float cos_dphi = cosf(phi_center - phi_pix);
            float cangdist = vz_zring + cos_dphi * sqrt_term;
            
            if (cangdist >= cosrad) {
                // Clamp ip for ring2nest_cuda (safety check)
                int ip_clamped = max(0, min(ring_pixels - 1, ip));
                int nest_pix = healpix_util::ring2nest_cuda(ir, ip_clamped, nside, order);
                
                // Bounds check on nest_pix
                int npix = 12 * nside * nside;
                if (nest_pix >= 0 && nest_pix < npix) {
                    if (myKey_touched != nullptr) {
                        // SAFETY: Ensure we don't write beyond allocated space
                        // In second call, off is from first call's prefix sum
                        // max_tiles is how many tiles were counted in first call
                        uint32_t max_tiles = tiles_touched[idx];
                        if (tile_num < max_tiles) {
                            myKey_touched[tile_num] = nest_pix;
                        }
                    }
                    tile_num++;
                }
            }
        }
    }
    
    // Warn if counts differ between first and second call (indicates numerical instability)
    if (myKey_touched != nullptr) {
        uint32_t old_num = tiles_touched[idx];
        if (old_num != tile_num) {
            // Don't update tiles_touched to avoid breaking prefix sum
            // Just use the smaller of the two to be safe
            printf("Warning: query_disc_rect tiles_touched mismatch for Gaussian %d: expected %d, got %d, radius %f\n", 
                   idx, old_num, tile_num, radius);
        }
    } else {
        // First call: just store the count
        tiles_touched[idx] = tile_num;
    }
}

// ============ Optimized version: skip healpix initialization (for second call) ============
// When healpix_gp is already initialized by the first call, use this version to skip init
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
    uint32_t* point_offsets)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P) return;

    const float MIN_RADIUS = 1e-7f;
    double r = double(radii[idx]);
    if (!isfinite(r) || r <= double(MIN_RADIUS)){
        return;  // tiles_touched already set to 0 by first call
    }
    
    uint32_t touch_num = tiles_touched[idx];
    if (touch_num == 0) return;  // Early exit if no tiles to touch
    
    double radius = radii[idx];
    bool inclusive = (fact != 0);
    float lon = lonlat_image[idx].x;
    float lat = lonlat_image[idx].y;

    float theta = M_PI / 2.0f - lat;
    float phi0 = lon + M_PI;
    
    uint32_t off = (idx == 0) ? 0 : point_offsets[idx - 1];
    int* myKey_touched = &tilesKey_touched[off];
    
    float2 ptg = {theta, phi0};
    normalize_pt(ptg);
    
    float cosrad = cos(radius);
    int order = round(log2(nside));
    int oplus = inclusive ? round(log2(fact)) : 0;
    int omax = order + oplus;
    int s_size = 12 + 6 * omax;

    // NOTE: Skip healpix_gp initialization - already done by first call!
    
    int2* my_stk = &d_stk_data[idx * s_size];
    int stack_size = 0;
    uint32_t tile_num = 0;
    
    for (int i = 0; i < 12; ++i) {
        my_stk[stack_size] = {11 - i, 0};
        stack_size++;
    }
    
    int stacktop = 0;
    int myoff = 0;
    
    while(stack_size > 0){
        int debug = 0;
        stack_size--;
        int2 current = my_stk[stack_size];
        int pix = current.x;
        int o_n = current.y;
        int o = idx * (omax + 1) + o_n;
        
        double z, phi;
        healpix_util::pix2zphiCUDA(pix, z, phi, healpix_gp[o]);
        
        float3 vptg = {sin(ptg.x)*cos(ptg.y), sin(ptg.x)*sin(ptg.y), cos(ptg.x)};
        double cangdist = vptg.z * z + cos(ptg.y - phi) * sqrt((1. - vptg.z * vptg.z) * (1. - z * z));
        
        if (cangdist > healpix_gp[o].crpdr){
            int zone = (cangdist < cosrad) ? 1 : ((cangdist <= healpix_gp[o].crmdr) ? 2 : 3);
            check_pixelCUDA(o_n, order, omax, zone, pix, my_stk, inclusive, stack_size, stacktop, tile_num, s_size, myoff, touch_num, myKey_touched, debug);
        }
    }
    
    // Verify consistency (can be removed in production)
    if (tile_num != touch_num){
        printf("Warning: tile count mismatch for Gaussian %d: expected %d, got %d\n", idx, touch_num, tile_num);
    }
}
