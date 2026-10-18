#include <cloth_common_glsl>
// Jacobi vertex-vertex pass. in0 positions, in1 sorted keys, 3D in0 bins.
uniform vec4 uBMin;
uniform vec4 uBMax;
uniform vec4 uHash;
uniform vec4 uAero;
layout(local_size_x = 8, local_size_y = 8) in;

void main()
{
	ivec2 size = ivec2(uTDOutputInfo.res.zw);
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	if (coord.x >= size.x || coord.y >= size.y)
		return;
	vec4 cur = texelFetch(sTD2DInputs[0], coord, 0);
	int id = coord.x + coord.y * size.x;
	vec3 p = clothSelf(cur.xyz, cur.w, id, size, sTD2DInputs[0], sTD2DInputs[1], sTD3DInputs[0],
		uBMin.xyz, uBMax.xyz, max(int(uHash.x + 0.5), 1), uAero.w);
	TDImageStoreOutput(0, ivec3(coord, 0), vec4(p, cur.w));
}
