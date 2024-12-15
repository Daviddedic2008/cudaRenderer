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
#define dot(vec3_v1, vec3_v2) (vec3_v1.x * vec3_v2.x + vec3_v1.y * vec3_v2.y + vec3_v1.z * vec3_v2.z)
#define matrix2D_eval(float_a , float_b, float_c, float_d) (float_a*float_d - float_b*float_c)
#define matgnitude(vec3_a) (sqrtf(dot(vec3_a, vec3_a)))

// Define the vec3 struct
struct vec3 {
	float x, y, z;

	__host__ __device__ vec3() : x(0), y(0), z(0) {}
	__host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}

	inline __host__ __device__ vec3 operator+(const vec3& f) const {
		return vec3(x + f.x, y + f.y, z + f.z);
	}

	inline __host__ __device__ vec3 operator-(const vec3& f) const {
		return vec3(x - f.x, y - f.y, z - f.z);
	}

	inline __host__ __device__ vec3 operator*(float scalar) const {
		return vec3(x * scalar, y * scalar, z * scalar);
	}
};

// cross is more logical as its own function

inline __host__ __device__ vec3 cross(const vec3 v1, const vec3 v2) {
	vec3 ret;
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
	vec3 origin, direction;
	__host__ __device__ ray() : origin(vec3(0.0f, 0.0f, 0.0f)), direction(vec3(0.0f, 0.0f, 0.0f)){}
	__host__ __device__ ray(vec3 origin, vec3 direction) : origin(origin), direction(direction){}
};

struct triangle{
	vec3 p1, p2, p3;
	vec3 nv;
	vec3 sb21, sb31;
	float dot2121, dot2131, dot3131;
	bool unbounded;

	__host__ __device__ triangle() : p1(vec3(0.0f, 0.0f, 0.0f)), p2(vec3(0.0f, 0.0f, 0.0f)), p3(vec3(0.0f, 0.0f, 0.0f)){}

	__host__ __device__ triangle(vec3 P1, vec3 P2, vec3 P3, bool u) {
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
// init as chars to bypass restrictions on dynamic initialization
__device__ char triangles[num_triangles * sizeof(triangle)]; // all triangles(on global mem, so slow access)
__device__ char triangle_materials[num_triangles * sizeof(material)]; // materials corresponding to triangles
__device__ char rays[scr_w * scr_h * sizeof(ray)];

__constant__ char triangle_loader[triangles_per_load * sizeof(triangle)]; // for fast access from cached triangles

typedef struct {
	vec3 intersect;
	bool intersect_found;
}intersect_return;


union fbitwise {
	float f;
	unsigned int s;
};

// intersect funcs that will be put in kernel
__inline__ __device__ intersect_return find_closest_int(const triangle triangles_loaded[triangles_per_load], const ray r, const int tris_read) {
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
		vec3 temp_sub = triangles_loaded[t].p1 - r.origin;
		temp_sub = r.direction * __fdividef(dot(triangles_loaded[t].nv, temp_sub), disc.f);// fast division since fastmath doesnt work on my system for some reason
		ret.intersect = r.origin + temp_sub;
		vec3 v2 = ret.intersect - triangles_loaded[t].p1;
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
		((triangle*)triangle_loader)[id] = ((triangle*)triangles)[id + pass * triangles_per_load];
	}
	__syncthreads();
	int tbd = num_triangles - pass * triangles_per_load;
	return find_closest_int((triangle*)triangle_loader, r, (tbd < triangles_per_load) * (tbd - triangles_per_load) + triangles_per_load);
}