#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <iostream>

// macros for sizes
#define num_triangles 1
#define scr_w 512
#define scr_h 512
#define triangles_per_load 256

// macros to replace functions
#define dot(float3_v1, float3_v2) (float3_v1.x * float3_v2.x + float3_v1.y * float3_v2.y + float3_v1.z * float3_v2.z)
#define matrix2D_eval(float_a , float_b, float_c, float_d) (float_a*float_d - float_b*float_c)
#define matgnitude(float3_a) (sqrtf(dot(float3_a, float3_a)))

// Define the float3 struct
struct float3 {
	float x, y, z;

	__host__ __device__ float3() : x(0), y(0), z(0) {}
	__host__ __device__ float3(float x, float y, float z) : x(x), y(y), z(z) {}

	inline __host__ __device__ float3 operator+(const float3& f) const {
		return float3(x + f.x, y + f.y, z + f.z);
	}

	inline __host__ __device__ float3 operator-(const float3& f) const {
		return float3(x - f.x, y - f.y, z - f.z);
	}

	inline __host__ __device__ float3 operator*(float scalar) const {
		return float3(x * scalar, y * scalar, z * scalar);
	}
};

// cross is more logical as its own function

inline __host__ __device__ float3 cross(const float3 v1, const float3 v2) {
	float3 ret;
	ret.x = matrix2D_eval(v1.y, v1.z, v2.y, v2.z);
	ret.y = matrix2D_eval(v1.x, v1.z, v2.x, v2.z);
	ret.z = matrix2D_eval(v1.x, v1.y, v2.x, v2.y);
	return ret;
}


// structs
typedef struct {
	float r, g, b;
}color;

typedef struct {
	color c;
	float brightness, roughness;
}material;

struct ray{
	float3 origin, direction;
	__host__ __device__ ray() : origin(float3(0.0f, 0.0f, 0.0f)), direction(float3(0.0f, 0.0f, 0.0f)){}
	__host__ __device__ ray(float3 origin, float3 direction) : origin(origin), direction(direction){}
};

struct triangle{
	float3 p1, p2, p3;
	float3 nv;
	float3 sb21, sb31;
	float dot2121, dot2131, dot3131;
	bool unbounded;

	__host__ __device__ triangle(float3 P1, float3 P2, float3 P3, bool u) {
		p1 = P1;
		p2 = P2;
		p3 = P3;
		sb21 = p2 - p1;
		sb31 = p3 - p1;
		dot2121 = dot(sb21, sb21);
		dot2131 = dot(sb21, sb31);
		dot3131 = dot(sb31, sb31);
		nv = cross(sb21, sb31);
		unbounded = u;
	}
};

// global device arrs
__device__ triangle triangles[num_triangles]; // all triangles(on global mem, so slow access)
__device__ material triangle_materials[num_triangles]; // materials corresponding to triangles
__device__ ray rays[scr_w * scr_h];

__constant__ triangle triangle_loader[triangles_per_load]; // for fast access from cached triangles

typedef struct {
	float3 intersect;
	bool intersect_found;
}intersect_return;


union fbitwise {
	float f;
	unsigned int s;
};

// intersect funcs that will be put in kernel
__inline__ __device__ __host__  intersect_return find_closest_int(const triangle triangles_loaded[triangles_per_load], const ray r, const int tris_read) {
	intersect_return ret;
	ret.intersect_found = false;
	float closest_dist = -1.0f;
	unsigned int closest_ind;
	for (unsigned int t = 0; t < tris_read /*tris_read used bc not all triangles in array may be intialized*/; t++) {
		fbitwise disc;
		disc.f = dot(r.direction, triangles_loaded[t].nv);
		float dt = disc.s && 0x7FFFFFFF; // make float positive
		if (dt < 1e-16) { // check if the plane and ray are paralell enough to be ignored
			continue;
		}
		float3 temp_sub = triangles_loaded[t].p1 - r.origin;
		temp_sub = r.direction * __fdividef(dot(triangles_loaded[t].nv, temp_sub), disc.f);// fast division since fastmath doesnt work on my system for some reason
		ret.intersect = r.origin + temp_sub;
		float3 v2 = ret.intersect - triangles_loaded[t].p1;
		float dot02 = dot(triangles_loaded[t].sb21, v2);
		float dot12 = dot(triangles_loaded[t].sb31, v2);
		float u = (triangles_loaded[t].dot3131 * dot02 - triangles_loaded[t].dot2131 * dot12) * __fdividef(1.0f, (triangles_loaded[t].dot2121 * triangles_loaded[t].dot3131 - triangles_loaded[t].dot2131 * triangles_loaded[t].dot2131));
		float v = (triangles_loaded[t].dot2121 * dot12 - triangles_loaded[t].dot2131 * dot02) * __fdividef(1.0f, (triangles_loaded[t].dot2121 * triangles_loaded[t].dot3131 - triangles_loaded[t].dot2131 * triangles_loaded[t].dot2131));
		if ((((u < 0) || (v < 0) || (u + v > 1) || dot(temp_sub, r.direction) < 0.0f)) && !triangles_loaded[t].unbounded) { continue; }
		float new_dist = matgnitude(temp_sub);
		if (new_dist < closest_dist || !ret.intersect_found) {
			closest_dist = new_dist;
			closest_ind = t;
			ret.intersect_found = true;
		}
	}
	return ret;
}

// other stuff for kernel organization

inline __device__ intersect_return get_closest_intersect_in_load(const int pass, const ray r) {
	// wrapper, might be removed in future due to overhead
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < triangles_per_load) {
		triangle_loader[id] = triangles[id + pass * triangles_per_load];
	}
	__syncthreads();
	int tbd = num_triangles - pass * triangles_per_load;
	return find_closest_int(triangle_loader, r, (tbd < triangles_per_load) * (tbd - triangles_per_load) + triangles_per_load);
}