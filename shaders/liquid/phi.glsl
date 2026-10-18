#include <bus3d_glsl>
#include <liquid_common_glsl>
// Narrow-band liquid phi for the bus. Negative inside liquid, positive out to 3 cells, and the
// empty sentinel everywhere else so a real solid still wins the union. Alpha 1 marks "this is the
// liquid", which the classifier ignores when it reads the merged obstacle field back.
// in0 type.
// td: output3d
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

void main()
{
	ivec3 c = busCoord();
	ivec3 res = ivec3(busRes());
	if (!busLive(c))
		return;
	float h = busCell();
	if (lqKind(texelFetch(sTD3DInputs[0], c, 0).r) == LQ_LIQUID)
	{
		TDImageStoreOutput(0, c, vec4(-0.5 * h, 0.0, 0.0, 1.0));
		return;
	}
	float best = 1e4;
	for (int z = -3; z <= 3; z++)
		for (int y = -3; y <= 3; y++)
			for (int x = -3; x <= 3; x++)
			{
				ivec3 nb = c + ivec3(x, y, z);
				if (any(lessThan(nb, ivec3(0))) || any(greaterThanEqual(nb, res)))
					continue;
				if (lqKind(texelFetch(sTD3DInputs[0], nb, 0).r) != LQ_LIQUID)
					continue;
				best = min(best, length(vec3(x, y, z)) * h);
			}
	if (best < 1e3)
		TDImageStoreOutput(0, c, vec4(best, 0.0, 0.0, 1.0));
	else
		TDImageStoreOutput(0, c, vec4(BUS_EMPTY_DISTANCE, 0.0, 0.0, 0.0));
}
