#include <bus3d_glsl>
#include <liquid_common_glsl>
// A p = -laplacian(p) on liquid cells, the SPD operator MGPCG inverts.
// Air is Dirichlet (contribution 0). Solid and a closed wall mirror pressure.
// in0 pressure, in1 type.
// td: output3d
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

int kindAt(ivec3 c, ivec3 res)
{
	if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, res)))
		return -1;
	return lqKind(texelFetch(sTD3DInputs[1], c, 0).r);
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
	float pc = texelFetch(sTD3DInputs[0], c, 0).r;
	float sum = pAt(c + ivec3(1, 0, 0), res, pc, ivec3(1, 0, 0)) + pAt(c - ivec3(1, 0, 0), res, pc, ivec3(-1, 0, 0))
	          + pAt(c + ivec3(0, 1, 0), res, pc, ivec3(0, 1, 0)) + pAt(c - ivec3(0, 1, 0), res, pc, ivec3(0, -1, 0))
	          + pAt(c + ivec3(0, 0, 1), res, pc, ivec3(0, 0, 1)) + pAt(c - ivec3(0, 0, 1), res, pc, ivec3(0, 0, -1));
	float h = busCell();
	// -(sum - 6p) / h^2
	TDImageStoreOutput(0, c, vec4((6.0 * pc - sum) / (h * h), 0.0, 0.0, 1.0));
}
