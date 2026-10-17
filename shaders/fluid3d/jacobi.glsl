#include <bus3d_glsl>
// One Jacobi iteration per pass of (sum - 6 p) / h^2 = rhs. Solids and closed sides mirror the
// centre (Neumann). An open side uses ghost pressure 0 (Dirichlet). in0 pressure, in1 rhs, in2 obstacles.
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

float pAt(ivec3 nb, ivec3 res, float pc, ivec3 dir)
{
	if (any(lessThan(nb, ivec3(0))) || any(greaterThanEqual(nb, res)))
		return busPressureGhost(dir, pc);
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
	if (busSolid(texelFetch(sTD3DInputs[2], c, 0)))
	{
		TDImageStoreOutput(0, c, vec4(0.0));
		return;
	}
	float pc = texelFetch(sTD3DInputs[0], c, 0).r;
	float sum = pAt(c + ivec3(1, 0, 0), res, pc, ivec3(1, 0, 0)) + pAt(c - ivec3(1, 0, 0), res, pc, ivec3(-1, 0, 0))
	          + pAt(c + ivec3(0, 1, 0), res, pc, ivec3(0, 1, 0)) + pAt(c - ivec3(0, 1, 0), res, pc, ivec3(0, -1, 0))
	          + pAt(c + ivec3(0, 0, 1), res, pc, ivec3(0, 0, 1)) + pAt(c - ivec3(0, 0, 1), res, pc, ivec3(0, 0, -1));
	float h = busCell();
	float rhs = texelFetch(sTD3DInputs[1], c, 0).r;
	TDImageStoreOutput(0, c, vec4((sum - h * h * rhs) / 6.0, 0.0, 0.0, 1.0));
}
