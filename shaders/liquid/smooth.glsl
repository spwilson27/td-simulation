#include <bus3d_glsl>
#include <liquid_common_glsl>
// Weighted Jacobi on liquid cells. Air contributes 0 (Dirichlet). Solid and a closed wall mirror
// pressure (Neumann). An open wall is ghost pressure 0. uOmega.x is 2/3 on the fine levels and 1
// on the coarsest solve. in0 pressure, in1 rhs, in2 type.
// td: output3d
uniform vec4 uOmega;
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

int kindAt(ivec3 c, ivec3 res)
{
	if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, res)))
		return -1;
	return lqKind(texelFetch(sTD3DInputs[2], c, 0).r);
}

float pAt(ivec3 nb, ivec3 res, float pc, ivec3 dir)
{
	if (any(lessThan(nb, ivec3(0))) || any(greaterThanEqual(nb, res)))
		return busPressureGhost(dir, pc);
	int k = kindAt(nb, res);
	if (k == LQ_LIQUID)
		return texelFetch(sTD3DInputs[0], nb, 0).r;
	if (k == LQ_AIR)
		return 0.0;
	return pc;
}

void main()
{
	ivec3 c = busCoord();
	ivec3 res = ivec3(busRes());
	if (!busLive(c))
		return;
	if (kindAt(c, res) != LQ_LIQUID)
	{
		TDImageStoreOutput(0, c, vec4(0.0));
		return;
	}
	float prev = texelFetch(sTD3DInputs[0], c, 0).r;
	float sum = pAt(c + ivec3(1, 0, 0), res, prev, ivec3(1, 0, 0)) + pAt(c - ivec3(1, 0, 0), res, prev, ivec3(-1, 0, 0))
	          + pAt(c + ivec3(0, 1, 0), res, prev, ivec3(0, 1, 0)) + pAt(c - ivec3(0, 1, 0), res, prev, ivec3(0, -1, 0))
	          + pAt(c + ivec3(0, 0, 1), res, prev, ivec3(0, 0, 1)) + pAt(c - ivec3(0, 0, 1), res, prev, ivec3(0, 0, -1));
	float h = busCell();
	float rhs = texelFetch(sTD3DInputs[1], c, 0).r;
	float omega = uOmega.x > 0.0 ? uOmega.x : (2.0 / 3.0);
	float guess = (sum - h * h * rhs) / 6.0;
	TDImageStoreOutput(0, c, vec4((1.0 - omega) * prev + omega * guess, 0.0, 0.0, 1.0));
}
