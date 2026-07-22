/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define BLOCK_SIZE_3D (BLOCK_X_3D * BLOCK_Y_3D)
#define NUM_WARPS (BLOCK_SIZE/32)
#define CLAMP_NUM 1e-12f

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ void getRect(const float2 p, int2 ext_rect, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - ext_rect.x) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - ext_rect.y) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + ext_rect.x + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + ext_rect.y + BLOCK_Y - 1) / BLOCK_Y)))
	};
}


__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

__forceinline__ __device__ bool omni_culling (
	const float3& p_orig,
	const float* viewmatrix,
	float3& p_view,
	float4& p_view_hom
)
{
	p_view = transformPoint4x3(p_orig, viewmatrix);

	float d = sqrt(p_view.x * p_view.x + p_view.y * p_view.y + p_view.z * p_view.z);
	p_view_hom.x = p_view.x;
	p_view_hom.y = p_view.y;
	p_view_hom.z = p_view.z;
	p_view_hom.w = d;
	if (d <= 0.2f)
	{
		return true;
	}

	return false;
	
}

__forceinline__ __device__ float2 p3d2lonlat(const float3& pt)
{
	// float inv_r = 1.0f / (sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z) + 0.0000001f);
	float r2 = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
	if (r2 < 1e-30f){
		float2 scr = {0.0f, 0.0f};
		return scr;
	}
	float rho = sqrtf(pt.x * pt.x + pt.z * pt.z);
	float lon = atan2f(pt.x, pt.z);
	// float lat = asinf(pt.y * inv_r);
	float lat = atan2f(pt.y, rho);

	if ( lon < -3.141592 ) {lon = -3.141592;}
	if ( lon > 3.141592 ) {lon = 3.141592;}
	if ( lat < -1.570796 ) {lat = -1.570796;}
	if ( lat > 1.570796 ) {lat = 1.570796;}

	float2 scr = {
		lon,
		lat
	};
	return scr;
}

__forceinline__ __device__ float2 point3ToLonlatScreen(const float4& pt_and_r)
{
	float inv_r = 1.0f / (pt_and_r.w + 0.0000001f);

	float lon = atan2f(pt_and_r.x, pt_and_r.z);
	float lat = asinf(pt_and_r.y * inv_r);

	float2 scr = {
		lon * M_1_PIf32,
		lat * M_2_PIf32
	};
	return scr;
}

// __forceinline__ __device__ float2 lonlat2Equipixel(float lon, float lat, int W, int H, float2& equi)
// {
//     if (lon < -M_PIf) lon = -M_PIf+0.000001f;
//     else if (lon > M_PIf) lon = M_PIf-0.000001f;

// 	if(lat < -M_PIf / 2.0f) lat = -M_PIf / 2.0f + 0.000001f;
// 	else if (lat > M_PIf / 2.0f) lat = M_PIf / 2.0f - 0.000001f;
    
//     float x = ((lon + M_PIf) / (2.0f * M_PIf)) * W;
    
//     float y = ((lat + M_PIf / 2.0f) / M_PIf) * H; 

    
//     equi.x = fmaxf(0.0f, fminf(x, W - 1.0f));
//     equi.y = fmaxf(0.0f, fminf(y, H - 1.0f));
    
//     return equi;
// }

// __forceinline__ __device__ float2 equipixel2Lonlat(float x, float y, int W, int H, float2& lonlat)
// {

//     float norm_x = x / W;
//     float norm_y = y / H;
    
//     float lon = norm_x * 2.0f * M_PIf - M_PIf;
//     float lat = norm_y * M_PIf - M_PIf / 2.0f;
    
//     if (lon < -M_PIf) lon = -M_PIf+0.000001f;
//     else if (lon > M_PIf) lon = M_PIf-0.000001f;

// 	if(lat < -M_PIf / 2.0f) lat = -M_PIf / 2.0f + 0.000001f;
// 	else if (lat > M_PIf / 2.0f) lat = M_PIf / 2.0f - 0.000001f;
    
//     lonlat.x = lon;
//     lonlat.y = lat;
    
//     return lonlat;
// }

__forceinline__ __device__ float wrap_pi(float x) {
    const float TWO_PI = 2.0f * M_PIf;
    x = fmodf(x + M_PIf, TWO_PI);
    if (x < 0.0f) x += TWO_PI;
    return x - M_PIf; // (-pi, pi]
}
__forceinline__ __device__ float clamp_lat(float v) {
    const float HALF_PI = 0.5f * M_PIf;
    return fminf(fmaxf(v, -HALF_PI), HALF_PI);
}
__forceinline__ __device__ float wrap_2pi_pos(float x) {
    const float TWO_PI = 2.0f * M_PIf;
    x = fmodf(x, TWO_PI);
    if (x < 0.0f) x += TWO_PI;
    return x; // [0,2pi)
}


// Equirectangular pixel center -> lon/lat
// Note: Input is integer pixel indices (ix, iy), internally using center (ix+0.5, iy+0.5)
__forceinline__ __device__ void equipixel2Lonlat_center(int ix, int iy, int W, int H, float2& lonlat) {
    float x = (float)ix + 0.5f;
    float y = (float)iy + 0.5f;
    float u = x / (float)W; // [0,1)
    float v = y / (float)H; // [0,1)
    float lon = (u * 2.0f - 1.0f) * M_PIf; // [-pi, pi)
    float lat = -(0.5f - v) * M_PIf;        // [pi/2 -> -pi/2]
    lon = wrap_pi(lon);
    lat = clamp_lat(lat);
    lonlat = {lon, lat};
}

// lon/lat -> equirectangular integer pixel indices (nearest neighbor, fully symmetric to above)
// Returns ix, iy, with longitude wrapping and latitude clamping
__forceinline__ __device__ void lonlat2Equipixel_index(float lon, float lat, int W, int H, int2& equi) {
    lon = wrap_pi(lon);
    lat = clamp_lat(lat);

    float u = (lon / M_PIf + 1.0f) * 0.5f; // [0,1)
    float v = 0.5f + (lat / M_PIf);
    // float v = 0.5f - (lat / M_PIf);

    // From consistent center definition: x_f = u*W - 0.5, nearest neighbor to grid center => floor(x_f + 0.5)
    float x_f = u * (float)W - 0.5f;
    float y_f = v * (float)H - 0.5f;

    int xi = (int)floorf(x_f + 0.5f);
    int yi = (int)floorf(y_f + 0.5f);

    // Longitude wrapping (periodic)
    if (xi < 0) xi = (xi % W + W) % W;
    else if (xi >= W) xi = xi % W;

    // Latitude clamping
    if (yi < 0) yi = 0;
    else if (yi >= H) yi = H - 1;

    equi = {xi, yi};
}

__forceinline__ __device__ void greatCircleDistance(float2 lonlat1, float2 lonlat2, float& rad){
    //Haversine formula
    float lon1 = lonlat1.x, lat1 = lonlat1.y;
    float lon2 = lonlat2.x, lat2 = lonlat2.y;
    float dlat = lat2 - lat1;
    float dlon = lon2 - lon1;
    float a = sinf(dlat * 0.5f) * sinf(dlat * 0.5f) + cosf(lat1) * cosf(lat2) * sinf(dlon * 0.5f) * sinf(dlon * 0.5f);
    rad = 2.0f * atan2f(sqrtf(a), sqrtf(1.0f - a));
}

__forceinline__ __device__ void computeBearing(float2 lonlat1, float2 lonlat2, float& bearing){
    float lon1 = lonlat1.x, lat1 = lonlat1.y;
    float lon2 = lonlat2.x, lat2 = lonlat2.y;
    float dlon = lon2 - lon1;
    float y = sinf(dlon) * cosf(lat2);
    float x = cosf(lat1) * sinf(lat2) - sinf(lat1) * cosf(lat2) * cosf(dlon);
    bearing = atan2f(y, x);
}

__forceinline__ __device__ float shortest_angular_difference(float angle1, float angle2){
	float diff = angle2 - angle1;
	while (diff <= -M_PIf) diff += 2.0f * M_PIf;
	while (diff > M_PIf) diff -= 2.0f * M_PIf;
	return diff;
}

__forceinline__ __device__ void fmoduloCUDA(float& v1, float& v2){
    if (v1 >= 0)
        v1 = (v1 < v2) ? v1 : fmodf(v1, v2);
    else {
        float tmp = fmodf(v1, v2) + v2;
        v1 = (tmp == v2) ? 0.0f : tmp;
    }
}

__forceinline__ __device__ void normalize_pt(float2& p){
    float theta = p.x;
    float phi = p.y;
    float t_pi = 2.0f * M_PI;
    const float epsilon = 1e-6;
    fmoduloCUDA(theta, t_pi);
    if (theta>M_PI + epsilon)
    {
        phi+=M_PI;
        theta=t_pi-theta;
    }
    fmoduloCUDA(phi, t_pi);
    p.x = theta; 
    p.y = phi;
}

__forceinline__ __device__ float3 clampCov22(const float3& cov, float eps_eig){
	float a = cov.x, b = cov.y, c = cov.z;
	float tr   = a + c;
    float diff = a - c;
    float disc = sqrtf(fmaxf(0.0f, diff*diff + 4.0f*b*b));
    float lam_max = 0.5f * (tr + disc);
    float lam_min = 0.5f * (tr - disc);

    float cs, sn;
    if (fabsf(b) > 1e-30f || fabsf(diff) > 1e-30f) {
        float two_theta = atan2f(2.0f * b, diff); // 2θ
        float theta = 0.5f * two_theta;
        cs = cosf(theta);
        sn = sinf(theta);
    } else {
        cs = 1.0f;
        sn = 0.0f;
    }
    // clamp
    lam_max = fmaxf(lam_max, eps_eig);
    lam_min = fmaxf(lam_min, eps_eig);
    // reconstruct Σ̃ = Q diag(λmax, λmin) Q^T
    // Q = [ [cs, -sn], [sn, cs] ]
    float cs2 = cs*cs, sn2 = sn*sn, cs_sn = cs*sn;
    float S00 = lam_max*cs2 + lam_min*sn2;
    float S01 = (lam_max - lam_min)*cs_sn;
    float S11 = lam_max*sn2 + lam_min*cs2;

	float3 clamped_cov = {S00, S01, S11};
    return clamped_cov;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif
