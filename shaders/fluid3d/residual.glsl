#include <bus3d_glsl>
// rhs - Laplacian(p), for the pressure benchmark. in0 pressure, in1 rhs, in2 obstacles.
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

float pAt(ivec3 nb, ivec3 res, float pc)
{
	if (any(lessThan(nb, ivec3(0))) || any(greaterThanEqual(nb, res)))
		return pc;
	if (busSolid(texelFetch(sTD3DInputs[2], nb, 0)))
		return pc;
	return texelFetch(sTD3DInputs[0], nb, 0).r;
}

void main()
{
	ivec3 c = busCoord();
	ivec3 res = ivec3(busRes());
	if (!busLive(c))
		return;
	float pc = texelFetch(sTD3DInputs[0], c, 0).r;
	float sum = pAt(c + ivec3(1, 0, 0), res, pc) + pAt(c - ivec3(1, 0, 0), res, pc)
	          + pAt(c + ivec3(0, 1, 0), res, pc) + pAt(c - ivec3(0, 1, 0), res, pc)
	          + pAt(c + ivec3(0, 0, 1), res, pc) + pAt(c - ivec3(0, 0, 1), res, pc);
	float h = busCell();
	float rhs = texelFetch(sTD3DInputs[1], c, 0).r;
	float r = rhs - (sum - 6.0 * pc) / (h * h);
	if (busSolid(texelFetch(sTD3DInputs[2], c, 0)))
		r = 0.0;
	TDImageStoreOutput(0, c, vec4(r, 0.0, 0.0, 1.0));
}
