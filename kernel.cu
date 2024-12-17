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
#define passes_needed num_triangles / triangles_per_load + 1
#define fov 0.01f

// tracer related macros
#define num_bounces 1
#define num_frames 1000
#define blur 0.0f

// macros for gpu params
#define threads_main 256
#define blocks_main scr_w * scr_h / threads_main

// macros to replace functions
#define dot(vec3_v1, vec3_v2) (vec3_v1.x * vec3_v2.x + vec3_v1.y * vec3_v2.y + vec3_v1.z * vec3_v2.z)
#define matrix2D_eval(float_a , float_b, float_c, float_d) (float_a*float_d - float_b*float_c)
#define matgnitude(vec3_a) (sqrtf(dot(vec3_a, vec3_a)))

// too lazy to set up cudas rng so i use this bad one
inline __host__ __device__ unsigned int xorRand(int seed) {
	seed ^= seed << 13;
	seed ^= seed >> 17;
	seed ^= seed << 5;
	return seed;
}

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

	inline __host__ __device__ vec3 operator*(const float scalar) const {
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
struct color{
	float r, g, b;
	__host__ __device__ color(float R, float G, float B) : r(R), g(G), b(B){}

	inline __host__ __device__ color operator+(const color& f) const {
		return color(r + f.r, g + f.g, b + f.b);
	}

	inline __host__ __device__ color operator*(const float f) const {
		return color(r * f, g * f, b * f);
	}
};

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
__device__ char* screen_buffer;


// constant memory is generally faster than shared if all threads need to access it(which is the case here)
__constant__ char triangle_loader[triangles_per_load * sizeof(triangle)]; // for fast access from cached triangles

typedef struct {
	vec3 intersect;
	int triangle_index;
	float dist_from_origin;
	vec3 nv;
}intersect_return;


union fbitwise {
	float f;
	unsigned int s;
};

// intersect funcs that will be put in kernel
inline  __device__ intersect_return find_closest_int(const triangle triangles_loaded[triangles_per_load], const ray r, const int tris_read) {
	intersect_return ret;
	ret.triangle_index = -1;
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
		if (new_dist < closest_dist || (ret.triangle_index == -1)) {
			closest_dist = new_dist;
			closest_ind = t;
			ret.triangle_index = t;
			ret.nv = triangles_loaded[t].nv;
		}
	}
	ret.dist_from_origin = closest_dist;
	return ret;
}

// other stuff for kernel organization

inline __device__ intersect_return get_closest_intersect_in_load(const int pass, const ray r) {
	// wrapper, might be removed in future due to overhead
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	// load triangles into cached mem
	if (id < triangles_per_load) {
		((triangle*)triangle_loader)[id] = ((triangle*)triangles)[id + pass * triangles_per_load];
	}
	__syncthreads();

	int tbd = num_triangles - pass * triangles_per_load;
	return find_closest_int((triangle*)triangle_loader, r, (tbd < triangles_per_load) * (tbd - triangles_per_load) + triangles_per_load);
}

inline __device__ ray reflect_ray(ray r, vec3 nv, const vec3 intersect, const float random_strength, const unsigned int iteration) {
	// specular
	float dt = dot(r.direction, nv);
	nv =  nv * ((dt < 0.0f) * -1);
	dt = fabs(dt);
	r.direction = r.direction - (r.direction - nv * dt) * 2;
	r.direction = r.direction * -1;

	// random reflection
	unsigned int randx, randy, randz;
	randx = xorRand((threadIdx.x + blockIdx.x * blockDim.x) * iteration);
	randy = randx ^ randx >> 5;
	randz = randy ^ randy << 3;
	r.direction = r.direction * (1 - random_strength) + cross(vec3((randx % 1000) / 1000.0f, (randy % 1000) / 1000.0f, (randz % 1000) / 1000.0f), nv) * random_strength;
	return r;
}

// test kernels(not used)
/*
__device__ bool hitTri;
__global__ void test_int() {
	ray r = ray(vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f));
	hitTri = get_closest_intersect_in_load(0, r).intersect_found;
	printf("%d\n", hitTri);
}

void add_triangle_test() {
	char triangle2[sizeof(triangle)];
	((triangle*)triangle2)[0] = triangle(vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, 100.0f, 1.0f), vec3(100.0f, 100.0f, 1.0f), false);
	cudaMemcpyToSymbol(triangles, triangle2, sizeof(triangle2));
}
*/

inline __device__ ray initialize_rays(const int idx) {
	int x, y;
	x = idx % scr_w - scr_w / 2;
	y = idx / scr_w - scr_h / 2;
	return ray(vec3(x, y, 0.0f), vec3(x * fov, y * fov, 1.0f));
}

// reused file write func
inline FILE* open_file(const char* filename) {
	FILE* ret = fopen(filename, "w");
	if (ret == NULL) {
		printf("%s\n", "error opening file %s\n", filename);
		return NULL;
	}
	return ret;
}

void write_pixel_data_to_txt(const color* color_buffer) {
	unsigned char* pixels;
	FILE* f = open_file("colorReturnFile.txt");
	for (int l = 0; l < scr_w * scr_h; l++) {
		fprintf(f, "%f,%f,%f\n", color_buffer[l].r, color_buffer[l].g, color_buffer[l].b);
	}
	fclose(f);
}

// color stuff

inline __device__ void add_color(const int index, const color c) {
	((color*)screen_buffer)[index] = ((color*)screen_buffer)[index] + c;
}

inline __device__ void scale_color(const int index, const float f) {
	((color*)screen_buffer)[index] = ((color*)screen_buffer)[index] * f;
}

// kernel!

__global__ void updateKernel(const int iteration) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	// init ray at starting point
	ray r = initialize_rays(idx);

	intersect_return ret, temp;
	unsigned char num_hits = 0;
	color c = color(0.0f, 0.0f, 0.0f);
	for (int b = 0; b < num_bounces; b++) {
		int tri_id = -1;
		float cd = -1.0f;
		// get closest intersect from all triangles(no bvh for now)
		for (int p = 0; p < passes_needed; p++) {
			temp = get_closest_intersect_in_load(p, r);
			if (temp.triangle_index != -1 && (cd < 0.0f || temp.dist_from_origin < cd)) {
				ret = temp;
				cd = ret.dist_from_origin;
				tri_id = temp.triangle_index + p * triangles_per_load;
				++num_hits;
			}
		}
		if (tri_id == -1) {
			break;
		}
		// bounces ray and does color addition to buffer
		r = reflect_ray(r, ret.nv, ret.intersect, 0.0f, iteration);
		c = c + ((material*)triangle_materials)[tri_id].c;
	}
	// divide color by total num ints
	((color*)screen_buffer)[idx] = ((color*)screen_buffer)[idx] + (c * (1.0f / num_hits));
}

// copying func
void copyScreenBuffer(color* c) {
	cudaMemcpyFromSymbol(c, screen_buffer, sizeof(color) * scr_w * scr_h);
}

// tri cpu
void add_triangle(triangle t, int idx) {
	cudaMemcpyToSymbol(triangles, &t, sizeof(triangle), idx * sizeof(triangle));
}

int main() {
	cudaMalloc(&screen_buffer, sizeof(color) * scr_w * scr_h);
	cudaMemset(screen_buffer, 0, sizeof(color) * scr_w * scr_h);
	add_triangle(triangle(vec3(0.0f, 0.0f, 1.0f), vec3(100.0f, 0.0f, 1.0f), vec3(100.0f, 100.0f, 1.0f), false), 0);
	clock_t start, end;
	start = clock();
	for (int f = 0; f < num_frames; f++) {
		updateKernel << <threads_main, blocks_main >> > (f);
	}
	end = clock();
	cudaError_t e = cudaGetLastError();
	printf("kernel calls took %d miliseconds\n", end - start);
	printf("Kernel exited with error: %s\n", cudaGetErrorString(e));
}