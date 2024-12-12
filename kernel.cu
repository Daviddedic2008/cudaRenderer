
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string>

#define scr_w 512
#define scr_h 512

#define num_bounces 1
#define num_frames 100

__inline__ __device__ float3 init_float3(float x, float y, float z) {
	float3 ret;
	ret.x = x;
	ret.y = y;
	ret.z = z;
	return ret;
}

__inline__ __device__ float3 add_float3(float3 v1, float3 v2) {
	v1.x += v2.x;
	v1.y += v2.y;
	v1.z += v2.z;
	return v1;
}

__inline__ __device__ float3 sub_float3(float3 v1, float3 v2) {
	v1.x -= v2.x;
	v1.y -= v2.y;
	v1.z -= v2.z;
	return v1;
}

__inline__ __device__ float3 mult_float3(float3 v1, float3 v2) {
	v1.x *= v2.x;
	v1.y *= v2.y;
	v1.z *= v2.z;
	return v1;
}

__inline__ __device__ float3 scale_float3(float3 v1, float scl) {
	v1.x *= scl;
	v1.y *= scl;
	v1.z *= scl;
	return v1;
}

#define dot(float3_v1, float3_v2) (float3_v1.x * float3_v2.x + float3_v1.y * float3_v2.y + float3_v1.z * float3_v2.z)
#define matrix2D_eval(float_a , float_b, float_c, float_d) (float_a*float_d - float_b*float_c)
#define matgnitude(float3_a) (sqrtf(dot(float3_a, float3_a)))

__inline__ __device__ float3 cross_float3(float3 v1, float3 v2) {
	float3 ret;
	ret.x = matrix2D_eval(v1.y, v1.z, v2.y, v2.z);
	ret.y = matrix2D_eval(v1.x, v1.z, v2.x, v2.z);
	ret.z = matrix2D_eval(v1.x, v1.y, v2.x, v2.y);
	return ret;
}

__inline__ __device__ float3 normalize_float3(float3 v1) {
	float scl = 1 / sqrtf(dot(v1, v1));
	v1 = scale_float3(v1, scl);
	return v1;
}

__inline__ __device__ float3 invert_float3(float3 f) {
	f.x *= -1;
	f.y *= -1;
	f.z *= -1;
	return f;
}

__inline__ __device__ float3 norm_float3(float3 p1, float3 p2, float3 p3) {
	return cross_float3(sub_float3(p1, p2), sub_float3(p1, p3));
}

__inline__ __device__ float3 rand_float3(unsigned int seed) {
	float3 ret;
	seed ^= seed << 13;
	seed ^= seed >> 17;
	seed ^= seed << 5;
	ret.x = (seed % 10000) / 10000.0f;
	seed ^= seed * seed;
	ret.y = (seed % 10000) / 10000.0f;
	seed ^= seed * seed;
	ret.z = (seed % 10000) / 10000.0f;
	return ret;
}

__inline__ __device__ float3 rand_offset_float3(float3 v, float3 norm, float strength, unsigned int seed) {
	float3 rnd = rand_float3(seed);
	float3 axis = cross_float3(rnd, norm);
	seed ^= seed << 13;
	seed ^= seed >> 17;
	seed ^= seed << 5;
	float scl = (seed % 1000) / 1000.0f;
	axis = scale_float3(axis, scl * strength);
	v = scale_float3(v, 1.0f - strength);
	return add_float3(v, axis);
}

typedef struct {
	float r, g, b;
}color;

__device__ color device_color_buffer[scr_w * scr_h];
color color_buffer[scr_w * scr_h];

__device__ float max_brightness_buffer[scr_w * scr_h];

color ray_color_buffer[scr_w * scr_h];

__device__ __inline__ color init_color(float r, float g, float b) {
	color ret;
	ret.r = r;
	ret.g = g;
	ret.b = b;
	return ret;
}

typedef struct {
	color c;
	float brightness, roughness;
}material;

__inline__ __device__ material init_material(float r, float g, float b, float roughness, float brightness) {
	material ret;
	ret.c = init_color(r, g, b);
	ret.brightness = brightness;
	ret.roughness = roughness;
	return ret;
}

typedef struct {
	float3 origin, direction;
	float3 last_intersect;
	float last_dist;
	int last_triangle_index;
	bool hit_triangle;
	unsigned char num_intersects;
}ray;

__device__ ray rays[scr_w * scr_h];
#define fov 0.02f

__global__ void init_rays() {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int x, y;
	x = id % scr_w;
	y = id / scr_w;
	ray r;
	r.origin.x = x;
	r.origin.y = y;
	r.origin.z = 0.0f;
	r.direction.x = x * fov;
	r.direction.y = y * fov;
	r.direction.z = 1.0f;
	r.num_intersects = 0;
	device_color_buffer[id] = init_color(0.0f, 0.0f, 0.0f);
	max_brightness_buffer[id] = 0.0f;
	rays[id] = r;
}

__global__ void reset_rays() {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	rays[id].last_dist = -1.0f;
	rays[id].last_triangle_index = -1;
	rays[id].hit_triangle = false;
}

void init_rays_call() {
	init_rays << <1024, scr_w* scr_h / 1024 >> > ();
}

void reset_rays_call() {
	reset_rays << <1024, scr_w* scr_h / 1024 >> > ();
}

#pragma 
typedef struct {
	float3 p1, p2, p3;
	float3 nv;
	float3 sb21, sb31;
	float dot2121, dot2131, dot3131;
}triangle;

#define num_triangles 12
#define triangles_per_load 256

__device__ triangle triangles[num_triangles];
__device__ material triangle_materials[num_triangles];

__inline__ __device__ triangle init_triangle(float3 p1, float3 p2, float3 p3) {
	triangle ret;
	ret.p1 = p1;
	ret.p2 = p2;
	ret.p3 = p3;
	ret.nv = norm_float3(p1, p2, p3);
	ret.sb21 = sub_float3(p2, p1);
	ret.sb31 = sub_float3(p3, p1);
	ret.dot2121 = dot(ret.sb21, ret.sb21);
	ret.dot2131 = dot(ret.sb31, ret.sb21);
	ret.dot3131 = dot(ret.sb31, ret.sb31);
	return ret;
}

__global__ void init_cube(float3 p1, float l, int index, float r, float g, float b, float br, float roughness) {
	// Compute the 8 vertices of the cube
	float3 p2 = make_float3(p1.x + l, p1.y, p1.z);
	float3 p3 = make_float3(p1.x, p1.y - l, p1.z);
	float3 p4 = make_float3(p1.x + l, p1.y - l, p1.z);
	float3 p5 = make_float3(p1.x, p1.y, p1.z - l);
	float3 p6 = make_float3(p1.x + l, p1.y, p1.z - l);
	float3 p7 = make_float3(p1.x, p1.y - l, p1.z - l);
	float3 p8 = make_float3(p1.x + l, p1.y - l, p1.z - l);

	triangles[index] = init_triangle(p1, p3, p2);       // Front face
	triangles[index + 1] = init_triangle(p2, p3, p4);
	triangles[index + 2] = init_triangle(p5, p6, p7);   // Back face
	triangles[index + 3] = init_triangle(p6, p8, p7);
	triangles[index + 4] = init_triangle(p1, p5, p3);   // Left face
	triangles[index + 5] = init_triangle(p5, p7, p3);
	triangles[index + 6] = init_triangle(p2, p4, p6);   // Right face
	triangles[index + 7] = init_triangle(p6, p4, p8);
	triangles[index + 8] = init_triangle(p1, p2, p5);   // Top face
	triangles[index + 9] = init_triangle(p2, p6, p5);
	triangles[index + 10] = init_triangle(p3, p7, p4);  // Bottom face
	triangles[index + 11] = init_triangle(p7, p8, p4);
	for (int a = 0; a < 12; a++) {
		triangle_materials[index + a] = init_material(r, g, b, roughness, br);
	}
}

void init_cube_CPU(float3 p1, float l, int index, float r, float g, float b, float bright, float rough) {
	init_cube << <1, 1 >> > (p1, l, index, r, g, b, bright, rough);
}

__global__ void set_triangle_kernel(float3 p1, float3 p2, float3 p3, int index, float R, float G, float B, float b, float r) {
	triangles[index] = init_triangle(p1, p2, p3);
	triangle_materials[index] = init_material(R, G, B, r, b);
}

void set_triangle(float3 p1, float3 p2, float3 p3, int index) {
	set_triangle_kernel << <1, 1 >> > (p1, p2, p3, index, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f);
}

#define threads_bounce 512
#define blocks_bounce scr_w * scr_h / threads_bounce

__constant__ triangle tempLoader[triangles_per_load];

__global__ void bounceKernel(int iteration) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int passes = num_triangles / triangles_per_load + 1;
	ray r = rays[index];
	for (int p = 0; p < passes; p++) {
		int tid = p * triangles_per_load + index;
		if (tid < num_triangles && tid < triangles_per_load) {
			tempLoader[threadIdx.x] = triangles[tid];
		}
		__syncthreads();
		float tmp = p * triangles_per_load;
		
#pragma unroll
		for (int ti = 0; ti < triangles_per_load && (ti + tmp) < num_triangles; ti++) {
			triangle t = tempLoader[ti];
			float disc = dot(r.direction, t.nv);
			if (disc == 0.0f) { continue; }
			float3 temp_sub;
			temp_sub.x = t.p1.x - r.origin.x;
			temp_sub.y = t.p1.y - r.origin.y;
			temp_sub.z = t.p1.z - r.origin.z;
			float tmp_dt = __fdividef(dot(t.nv, temp_sub), disc);
			float3 intersect;
			intersect.x = r.origin.x + r.direction.x * tmp_dt;
			intersect.y = r.origin.y + r.direction.y * tmp_dt;
			intersect.z = r.origin.z + r.direction.z * tmp_dt;
			temp_sub.x = intersect.x - r.origin.x;
			temp_sub.y = intersect.y - r.origin.y;
			temp_sub.z = intersect.z - r.origin.z;
			float3 v2;
			v2.x = intersect.x - t.p1.x; 
			v2.y = intersect.y - t.p1.y;
			v2.z = intersect.z - t.p1.z;
			float dot02 = dot(t.sb21, v2);
			float dot12 = dot(t.sb31, v2);
			float invD = __fdividef(1.0f, (t.dot2121 * t.dot3131 - t.dot2131 * t.dot2131));
			float u = (t.dot3131 * dot02 - t.dot2131 * dot12) * invD;
			float v = (t.dot2121 * dot12 - t.dot2131 * dot02) * invD;
			tmp_dt = dot(temp_sub, r.direction);
			if ((u < 0) || (v < 0) || (u + v > 1) || tmp_dt < 0.0f) { continue; }
			float newDist = matgnitude(temp_sub);
			if (r.hit_triangle && (newDist >= r.last_dist)) {
				continue;
			}
			r.last_dist = newDist;
			r.last_triangle_index = ti;
			r.last_intersect = intersect;
			r.hit_triangle = true;
		}
	}
	if (!r.hit_triangle) { return; }
	r.num_intersects++;
	triangle t = triangles[r.last_triangle_index];
	material m = triangle_materials[r.last_triangle_index];
	float3 nv = t.nv;
	float tmp_d = dot(r.direction, nv);
	r.direction.x = -1 * r.direction.x - nv.x * 2 * tmp_d;
	r.direction.y = -1 * r.direction.y - nv.y * 2 * tmp_d;
	r.direction.z = -1 * r.direction.z - nv.z * 2 * tmp_d;
	r.direction = rand_offset_float3(r.direction, nv, m.roughness, index * iteration);
	r.origin = r.last_intersect;
	rays[index] = r;
	color c = device_color_buffer[index];
	c.r += m.c.r;
	c.g += m.c.g;
	c.b += m.c.b;
	device_color_buffer[index] = c;
	max_brightness_buffer[index] = max_brightness_buffer[index] < m.brightness ? m.brightness : max_brightness_buffer[index];
}

void call_bounce_kernel(int iteration) {
	bounceKernel << <threads_bounce, blocks_bounce >> > (iteration);
}

__global__ void div_colors_kernel() {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	ray r = rays[id];
	if (!r.hit_triangle) { return; }
	float mult = max_brightness_buffer[id] / r.num_intersects;
	device_color_buffer[id].r *= mult;
	device_color_buffer[id].g *= mult;
	device_color_buffer[id].b *= mult;
}

void div_colors() {
	div_colors_kernel << <1024, scr_w* scr_h / 1024 >> > ();
}

ray raysCPU[scr_w * scr_h];

void copy_rays_CPU() {
	cudaMemcpyFromSymbol(raysCPU, rays, sizeof(raysCPU));
}

void copy_colors_CPU() {
	cudaMemcpyFromSymbol(color_buffer, device_color_buffer, sizeof(color_buffer));
}



FILE* open_file(char* filename) {
	FILE* ret = fopen(filename, "w");
	if (ret == NULL) {
		printf("%s\n", "error opening file %s\n", filename);
		return NULL;
	}
	return ret;
}

void write_pixel_data_to_txt() {
	unsigned char* pixels;
	FILE* f = open_file("colorReturnFile.txt");
	for (int l = 0; l < scr_w*scr_h; l++) {
		fprintf(f, "%f,%f,%f\n", color_buffer[l].r, color_buffer[l].g, color_buffer[l].b);
	}
	fclose(f);
}

void cycleRays(int iteration) {
	init_rays_call();
	for (int t = 0; t < num_bounces; t++) {
		reset_rays_call();
		call_bounce_kernel(iteration);
	}
	div_colors();
}

int main() {
	clock_t start, end;
	cudaDeviceReset();
	init_cube_CPU(make_float3(100.0f, 200.0f, 10.0f), 100.0f, 0, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f);
	start = clock();
	for (int f = 0; f < num_frames; f++) {
		cycleRays(f);
	}
	cudaDeviceSynchronize();
	end = clock();
	printf("milis for call: %d\n", end - start);
	copy_rays_CPU();
	copy_colors_CPU();
	write_pixel_data_to_txt();
	cudaError_t progErr = cudaGetLastError();
	printf("program ended with err: %s\n", cudaGetErrorString(progErr));
}