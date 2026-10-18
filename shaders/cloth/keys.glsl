#include <cloth_common_glsl>
// Sort key: cell id, then particle index. Matches the bitonic compare in sort.glsl.
uniform vec4 uBMin;
uniform vec4 uBMax;
uniform vec4 uHash;
layout(local_size_x = 8, local_size_y = 8) in;

void main()
{
	ivec2 size = ivec2(uTDOutputInfo.res.zw);
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	if (coord.x >= size.x || coord.y >= size.y)
		return;
	int id = coord.x + coord.y * size.x;
	vec3 p = texelFetch(sTD2DInputs[0], coord, 0).xyz;
	int h = max(int(uHash.x + 0.5), 1);
	ivec3 cell = clothCell(p, uBMin.xyz, uBMax.xyz, h);
	TDImageStoreOutput(0, ivec3(coord, 0), vec4(float(clothLinear(cell, h)), float(id), 1.0, 1.0));
}
