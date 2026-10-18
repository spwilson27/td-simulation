#include <liquid_common_glsl>
// Sort key per particle: cell id, particle index, active flag. Inactive particles use a sentinel
// cell past every real cell so they form one run at the end of the sorted list.
uniform vec4 uBMin;
uniform vec4 uBMax;
uniform vec4 uGrid; // xyz resolution
layout(local_size_x = 8, local_size_y = 8) in;

void main()
{
	ivec2 size = ivec2(uTDOutputInfo.res.zw);
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	if (coord.x >= size.x || coord.y >= size.y)
		return;
	int id = coord.x + coord.y * size.x;
	vec4 p = texelFetch(sTD2DInputs[0], coord, 0);
	ivec3 res = max(ivec3(uGrid.xyz + 0.5), ivec3(1));
	vec3 h = (uBMax.xyz - uBMin.xyz) / vec3(res);
	float cell = 10000000.0;
	float alive = 0.0;
	if (p.a > 0.0)
	{
		ivec3 c = ivec3(floor((p.xyz - uBMin.xyz) / h));
		c = clamp(c, ivec3(0), res - ivec3(1));
		cell = float(lqLinear(c, res));
		alive = 1.0;
	}
	TDImageStoreOutput(0, ivec3(coord, 0), vec4(cell, float(id), alive, 1.0));
}
