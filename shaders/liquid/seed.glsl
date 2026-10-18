#include <liquid_common_glsl>
// One texel per particle. uWrite.x < 0.5 writes position.xyz + mass, otherwise velocity.
// uScene.x: 0 dam break, 1 still tank, 2 uniform-velocity block (the transfer check).
// uScene.y spacing, uScene.z density, uScene.w transfer speed.
// Dead texels (past the lattice count) carry mass 0 and sort to the end of the bins.
uniform vec4 uBMin;
uniform vec4 uBMax;
uniform vec4 uScene;
uniform vec4 uWrite;
uniform vec4 uBox0; // xyz min as a fraction of the domain, w > 0.5 enables the box
uniform vec4 uBox1; // xyz max fraction
layout(local_size_x = 8, local_size_y = 8) in;

void main()
{
	ivec2 size = ivec2(uTDOutputInfo.res.zw);
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	if (coord.x >= size.x || coord.y >= size.y)
		return;
	int id = coord.x + coord.y * size.x;
	vec3 bmin = uBMin.xyz;
	vec3 bmax = uBMax.xyz;
	vec3 box0 = bmin;
	vec3 box1 = bmax;
	float spacing = max(uScene.y, 1e-4);
	vec3 vel = vec3(0.0);
	float mode = uScene.x;
	if (mode > 1.5)
	{
		box0 = mix(bmin, bmax, 0.25);
		box1 = mix(bmin, bmax, 0.75);
		vel = vec3(uScene.w, 0.0, 0.0);
	}
	else if (uBox0.w > 0.5)
	{
		box0 = mix(bmin, bmax, uBox0.xyz);
		box1 = mix(bmin, bmax, uBox1.xyz);
	}
	else if (mode < 0.5)
	{
		box1 = bmin + (bmax - bmin) * vec3(0.32, 0.62, 0.96);
		box0 = bmin + (bmax - bmin) * vec3(0.02, 0.02, 0.02);
	}
	else if (mode < 1.5)
	{
		box1 = bmin + (bmax - bmin) * vec3(0.98, 0.45, 0.98);
		box0 = bmin + (bmax - bmin) * vec3(0.02, 0.02, 0.02);
	}
	else
	{
		box0 = mix(bmin, bmax, 0.25);
		box1 = mix(bmin, bmax, 0.75);
	}
	ivec3 n = max(ivec3(floor((box1 - box0) / spacing + 1e-4)), ivec3(1));
	int count = n.x * n.y * n.z;
	if (id >= count)
	{
		TDImageStoreOutput(0, ivec3(coord, 0), uWrite.x < 0.5 ? vec4(bmin - vec3(1.0), 0.0) : vec4(0.0));
		return;
	}
	int x = id - (id / n.x) * n.x;
	int yz = id / n.x;
	int y = yz - (yz / n.y) * n.y;
	int z = yz / n.y;
	vec3 pos = box0 + (vec3(x, y, z) + 0.5) * spacing;
	float mass = uScene.z * spacing * spacing * spacing;
	if (uWrite.x < 0.5)
		TDImageStoreOutput(0, ivec3(coord, 0), vec4(pos, mass));
	else
		TDImageStoreOutput(0, ivec3(coord, 0), vec4(vel, 1.0));
}
