#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include "bmpwriter.h"
#include <vector>
#include <fstream>
#include <sstream>

// macros for sizes
#define num_triangles 36
#define scr_w 512
#define scr_h 512
#define triangles_per_load 256
#define passes_needed num_triangles / triangles_per_load + 1
#define fov 0.005f

// tracer related macros
#define num_bounces 4
#define num_frames 100
#define blur 0.01f
#define smoothing_constant 0.4

// macros for gpu params
#define threads_main 512
#define blocks_main scr_w * scr_h / threads_main

// macros to replace functions
#define dot(vec3_v1, vec3_v2) (vec3_v1.x * vec3_v2.x + vec3_v1.y * vec3_v2.y + vec3_v1.z * vec3_v2.z)
#define matrix2D_eval(float_a , float_b, float_c, float_d) (float_a*float_d - float_b*float_c)
#define matgnitude(vec3_a) (sqrtf(dot(vec3_a, vec3_a)))

// too lazy to set up cudas rng so i use this bad one
inline __host__ __device__ long int xorRand(unsigned int seed) {
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

	inline __host__ __device__ vec3 normalize() {
		const float scl = matgnitude((*this));
		return vec3(x / scl, y / scl, z / scl);
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

struct material {
	color c;
	float brightness, roughness;

	__host__ __device__ material() : c(color(0.0f, 0.0f, 0.0f)){}

	__host__ __device__ material(color C, float B, float rough) : c(C), brightness(B), roughness(rough){}
};

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
		nv = cross(sb21, sb31).normalize();
		unbounded = u;
	}
};

// global device arrs
// init as chars to bypass restrictions on dynamic initialization
__device__ char triangles[num_triangles * sizeof(triangle)]; // all triangles(on global mem, so slow access)
__device__ char triangle_materials[num_triangles * sizeof(material)]; // materials corresponding to triangles
__device__ char screen_buffer[scr_w * scr_h * sizeof(color)];

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
		const float dt = disc.s && 0x7FFFFFFF; // make float positive
		if (dt <= 1e-10) { // check if the plane and ray are paralell enough to be ignored
			continue;
		}
		vec3 temp_sub = triangles_loaded[t].p1 - r.origin;
		temp_sub = r.direction * __fdividef(dot(triangles_loaded[t].nv, temp_sub), disc.f);// fast division since fastmath doesnt work on my system for some reason
		ret.intersect = r.origin + temp_sub;
		const vec3 v2 = ret.intersect - triangles_loaded[t].p1;
		const float dot02 = dot(triangles_loaded[t].sb21, v2);
		const float dot12 = dot(triangles_loaded[t].sb31, v2);
		const float disc2 = (triangles_loaded[t].dot2121 * triangles_loaded[t].dot3131 - triangles_loaded[t].dot2131 * triangles_loaded[t].dot2131);
		if (disc2 == 0.0f) { continue; }
		const float u = (triangles_loaded[t].dot3131 * dot02 - triangles_loaded[t].dot2131 * dot12) * __fdividef(1.0f, disc2);
		const float v = (triangles_loaded[t].dot2121 * dot12 - triangles_loaded[t].dot2131 * dot02) * __fdividef(1.0f, disc2);
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

__constant__ char triangle_loader[triangles_per_load * sizeof(triangle)];

inline __device__ intersect_return get_closest_intersect_in_load(const int pass, const ray r) {
	// wrapper, might be removed in future due to overhead
	const int id = threadIdx.x + blockIdx.x * blockDim.x;

	// load triangles into cached mem
	// constant memory is generally faster than shared if all threads need to access it(which is the case here) // for fast access from cached triangles
	if (id < num_triangles && id < triangles_per_load) {
		((triangle*)triangle_loader)[threadIdx.x] = ((triangle*)triangles)[id + pass * triangles_per_load];
	}
	__syncthreads();

	const int tbd = num_triangles - pass * triangles_per_load;
	return find_closest_int((triangle*)triangle_loader, r, (tbd < triangles_per_load) * (tbd - triangles_per_load) + triangles_per_load);
}

inline __device__ ray reflect_ray(ray r, vec3 nv, const vec3 intersect, const float random_strength, const unsigned int iteration) {
	// specular
	const float dt = dot(r.direction, nv);
	//nv = nv * -1 * signbit(dt);
	//dt = fabs(dt);
	const vec3 dir = (r.direction - (r.direction - nv * dt) * 2) * -1;

	// random reflection
	const unsigned int randx = xorRand((threadIdx.x + blockIdx.x * blockDim.x) * (iteration+1) + 1223);
	const unsigned int randy = xorRand(randx);
	const unsigned int randz = xorRand(randy);
	const vec3 ran = vec3((randx % 1000) / 999.0f - 0.5f, (randy % 1000) / 999.0f - 0.5f, (randz % 1000) / 999.0f - 0.5f);
	r.direction = ((dir * (1.0f - random_strength)) + ran * random_strength);
	return r;
}

inline __device__ ray initialize_rays(const int idx, const int iteration) {
	float x, y;
	x = idx % scr_w - scr_w / 2;
	y = idx / scr_w - scr_h / 2;
	const int rx = xorRand(idx * iteration);
	const int ry = xorRand(rx);
	const vec3 rv = vec3((rx % 1000) / 999.0f - 0.5f, (ry % 1000) / 999.0f - 0.5f, 0.0f) * blur + vec3(x * fov, y * fov, 1.0f).normalize();
	return ray(vec3(x, y, 0.0f), rv);
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
	ray r = initialize_rays(idx, iteration);

	intersect_return ret;
	unsigned char num_hits = 0;
	color c = color(0.0f, 0.0f, 0.0f);
	material m;
	for (int b = 0; b < num_bounces; b++) {
		int tri_id = -1;
		float cd = -1.0f;
		// get closest intersect from all triangles(no bvh for now)
		for (int p = 0; p < passes_needed; p++) {
			const intersect_return temp = get_closest_intersect_in_load(p, r);
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
		m = ((material*)triangle_materials)[tri_id];
		r = reflect_ray(r, ret.nv, ret.intersect, m.roughness, iteration);
		c = m.c * (smoothing_constant) + c * (1.0f-smoothing_constant);
		if (m.brightness > 0.0f) {
			break;
		}
	}
	if (num_hits == 0) { return; }

	// divide color by total num ints
	((color*)screen_buffer)[idx] = ((color*)screen_buffer)[idx] + c * m.brightness;
}

// copying func
void copyScreenBuffer(color* c) {
	cudaMemcpyFromSymbol(c, screen_buffer, sizeof(color) * scr_w * scr_h);
}

// tri cpu
void add_triangle(triangle t, int idx, material m) {
	cudaMemcpyToSymbol(triangles, &t, sizeof(triangle), idx * sizeof(triangle));
	cudaMemcpyToSymbol(triangle_materials, &m, sizeof(material), idx * sizeof(material));
}

void initialize_cube(float side_length, vec3 origin, material m, int idx) {
	vec3 vertices[] = {
		vec3(origin.x, origin.y, origin.z), vec3(origin.x + side_length, origin.y, origin.z), vec3(origin.x + side_length, origin.y + side_length, origin.z), vec3(origin.x, origin.y + side_length, origin.z), // Bottom vertices
		vec3(origin.x, origin.y, origin.z + side_length), vec3(origin.x + side_length, origin.y, origin.z + side_length), vec3(origin.x + side_length, origin.y + side_length, origin.z + side_length), vec3(origin.x, origin.y + side_length, origin.z + side_length)  // Top vertices
	};

	add_triangle(triangle(vertices[0], vertices[1], vertices[2], false), idx, m);
	add_triangle(triangle(vertices[0], vertices[2], vertices[3], false), idx+1, m);

	add_triangle(triangle(vertices[4], vertices[5], vertices[6], false), idx+2, m);
	add_triangle(triangle(vertices[4], vertices[6], vertices[7], false), idx+3, m);

	add_triangle(triangle(vertices[0], vertices[1], vertices[5], false), idx+4, m);
	add_triangle(triangle(vertices[0], vertices[5], vertices[4], false), idx+5, m);

	add_triangle(triangle(vertices[2], vertices[3], vertices[7], false), idx+6, m);
	add_triangle(triangle(vertices[2], vertices[7], vertices[6], false), idx+7, m);

	add_triangle(triangle(vertices[0], vertices[3], vertices[7], false), idx+8, m);
	add_triangle(triangle(vertices[0], vertices[7], vertices[4], false), idx+9, m);

	add_triangle(triangle(vertices[1], vertices[2], vertices[6], false), idx+10, m);
	add_triangle(triangle(vertices[1], vertices[6], vertices[5], false), idx+11, m);
}

void initialize_cubeLight(float side_length, vec3 origin, material m1, material m2, int idx) {
	vec3 vertices[] = {
		vec3(origin.x, origin.y, origin.z), vec3(origin.x + side_length, origin.y, origin.z), vec3(origin.x + side_length, origin.y + side_length, origin.z), vec3(origin.x, origin.y + side_length, origin.z), // Bottom vertices
		vec3(origin.x, origin.y, origin.z + side_length), vec3(origin.x + side_length, origin.y, origin.z + side_length), vec3(origin.x + side_length, origin.y + side_length, origin.z + side_length), vec3(origin.x, origin.y + side_length, origin.z + side_length)  // Top vertices
	};

	// Bottom
	add_triangle(triangle(vertices[0], vertices[1], vertices[2], false), idx, m1);
	add_triangle(triangle(vertices[0], vertices[2], vertices[3], false), idx + 1, m1);

	// Top (roof)
	add_triangle(triangle(vertices[3], vertices[2], vertices[6], false), idx + 2, m1);
	add_triangle(triangle(vertices[3], vertices[6], vertices[7], false), idx + 3, m1);

	// Sides
	add_triangle(triangle(vertices[0], vertices[1], vertices[5], false), idx + 4, m2);
	add_triangle(triangle(vertices[0], vertices[5], vertices[4], false), idx + 5, m2);

	add_triangle(triangle(vertices[2], vertices[1], vertices[5], false), idx + 6, m1);
	add_triangle(triangle(vertices[2], vertices[5], vertices[6], false), idx + 7, m1);

	add_triangle(triangle(vertices[0], vertices[3], vertices[7], false), idx + 8, m1);
	add_triangle(triangle(vertices[0], vertices[7], vertices[4], false), idx + 9, m1);

	add_triangle(triangle(vertices[4], vertices[5], vertices[6], false), idx + 10, m1);
	add_triangle(triangle(vertices[4], vertices[6], vertices[7], false), idx + 11, m1);
}



bool read_stl(const std::string& filename, material m, int start_idx) {
	std::ifstream file(filename, std::ios::binary);
	if (!file) {
		std::cerr << "Failed to open file: " << filename << std::endl;
		return false;
	}

	// Read the 80-byte header (skip it)
	file.seekg(80, std::ios::beg);

	// Read the number of triangles in the STL file
	unsigned int numTriangles;
	file.read(reinterpret_cast<char*>(&numTriangles), sizeof(numTriangles));

	// Read each triangle
	for (unsigned int i = 0; i < numTriangles; ++i) {
		// Skip normal vector (3 floats)
		float normal[3];
		file.read(reinterpret_cast<char*>(&normal), sizeof(normal));

		// Read the 3 vertices of the triangle
		triangle t;
		file.read(reinterpret_cast<char*>(&t.p1.x), sizeof(float));  // p1.x
		file.read(reinterpret_cast<char*>(&t.p1.y), sizeof(float));  // p1.y
		file.read(reinterpret_cast<char*>(&t.p1.z), sizeof(float));  // p1.z

		file.read(reinterpret_cast<char*>(&t.p2.x), sizeof(float));  // p2.x
		file.read(reinterpret_cast<char*>(&t.p2.y), sizeof(float));  // p2.y
		file.read(reinterpret_cast<char*>(&t.p2.z), sizeof(float));  // p2.z

		file.read(reinterpret_cast<char*>(&t.p3.x), sizeof(float));  // p3.x
		file.read(reinterpret_cast<char*>(&t.p3.y), sizeof(float));  // p3.y
		file.read(reinterpret_cast<char*>(&t.p3.z), sizeof(float));  // p3.z

		// Skip attribute byte count (2 bytes)
		unsigned short attribute;
		file.read(reinterpret_cast<char*>(&attribute), sizeof(attribute));

		// Call add_triangle with the triangle and index
		add_triangle(t, start_idx + i, m);
	}

	file.close();
	return true;
}

// Function to read an ASCII STL file and add triangles
bool read_ascii_stl(const std::string& filename, material m, int start_idx) {
	std::ifstream file(filename);
	if (!file) {
		std::cerr << "Failed to open file: " << filename << std::endl;
		return false;
	}

	std::string line;
	int idx = start_idx;
	while (std::getline(file, line)) {
		if (line.find("facet normal") != std::string::npos) {
			// Skip the normal line
			std::getline(file, line);

			// Read the 3 vertices of the triangle
			triangle t;
			for (int i = 0; i < 3; ++i) {
				std::getline(file, line);
				std::stringstream ss(line);
				std::string temp;
				ss >> temp >> t.p1.x >> t.p1.y >> t.p1.z;
			}

			// Skip the endfacet line
			std::getline(file, line);

			// Call add_triangle with the triangle and index
			add_triangle(t, idx++, m);
		}
	}

	file.close();
	return true;
}

// The wrapper function to determine the file type (binary or ASCII) and process the STL file accordingly
bool process_stl_file(const std::string& filename, material m, int start_idx) {
	std::ifstream file(filename, std::ios::binary);
	if (!file) {
		std::cerr << "Failed to open file: " << filename << std::endl;
		return false;
	}

	char header[5];
	file.read(header, 5);
	file.close();

	// Check if file is binary or ASCII based on the header
	if (header[0] == 's' && header[1] == 'o' && header[2] == 'l' && header[3] == 'i' && header[4] == 'd') {
		// Likely ASCII STL, so use the ASCII parser
		return read_ascii_stl(filename, m, start_idx);
	}
	else {
		// Binary STL, so use the binary parser
		return read_stl(filename, m, start_idx);
	}
}




// zero kernel
__global__ void zeroBuffer() {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	((color*)screen_buffer)[idx] = color(0.0f, 0.0f, 0.0f);
}

// div colors stuff
__global__ void divBuffer() {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	color c = ((color*)screen_buffer)[idx] * (1.0f/num_frames);
	c.r = c.r > 1.0f ? 1.0f : c.r;
	c.g = c.g > 1.0f ? 1.0f : c.g;
	c.b = c.b > 1.0f ? 1.0f : c.b;
	((color*)screen_buffer)[idx] = c;
}

int main() {
	zeroBuffer << <256, scr_w* scr_h / 256 >> > ();
	initialize_cube(512.0f, vec3(-256.0f, -256.0f, -1.0f), material(color(0.0f, 1.0f, 0.0f), 0.0f, 0.4f), 0);
	initialize_cube(100.0f, vec3(-200.0f, 130.0f, 100.0f), material(color(1.0f, 0.0f, 0.0f), 8.0f, 0.0f), 12);
	initialize_cube(100.0f, vec3(100.0f, -100.0f, 100.0f), material(color(0.0f, 0.0f, 1.0f), 3.0f, 0.0f), 24);
	//add_triangle(triangle(vec3(-11.0f, 82.0f, 3.0f), vec3(-12.0f, 80.0f, 7.0f), vec3(-3.0f, 88.0f, 7.0f), false), 13, material(color(1.0f, 1.0f, 1.0f), 1.0f, 0.0f));

	//readStlModelAndAddTriangles(, material(color(1.0f, 1.0f, 1.0f), 1.0f, 0.0f));
	//process_stl_file("C:\\Users\\david\\Downloads\\pythonAndModels\\Knight.stl", material(color(1.0f, 1.0f, 1.0f), 1.0f, 0.0f), 0);
	clock_t start, end;
	start = clock();
	for (int f = 0; f < num_frames; f++) {
		updateKernel << <threads_main, blocks_main >> > (f);
	}
	cudaDeviceSynchronize();
	end = clock();
	divBuffer << <256, scr_w* scr_h / 256 >> > ();
	color* sc = (color*)malloc(sizeof(color) * scr_w * scr_h);
	copyScreenBuffer(sc);
	cudaDeviceSynchronize();
	saveBMP("out.bmp", scr_w, scr_h, sc);
	cudaError_t e = cudaGetLastError();
	printf("kernel calls took %d miliseconds\n", end - start);
	printf("Kernel exited with error: %s\n", cudaGetErrorString(e));
	free(sc);
	cudaFree(screen_buffer);
}