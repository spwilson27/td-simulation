#include <liquid_common_glsl>
#include <bus3d_glsl>
// Coarse cell type. Solid wins when at least half of the children are solid; otherwise any liquid
// child keeps the cell liquid so the free surface does not vanish between levels.
// in0 fine type.
// td: output3d
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

void main()
{
	ivec3 c = busCoord();
	ivec3 coarse = ivec3(busRes());
	if (!busLive(c))
		return;
	ivec3 fine = textureSize(sTD3DInputs[0], 0);
	ivec3 base = c * 2;
	if (fine != coarse * 2)
		base = c * fine / max(coarse, ivec3(1));
	int solids = 0;
	int liquids = 0;
	int seen = 0;
	for (int z = 0; z < 2; z++)
		for (int y = 0; y < 2; y++)
			for (int x = 0; x < 2; x++)
			{
				ivec3 fc = base + ivec3(x, y, z);
				if (any(lessThan(fc, ivec3(0))) || any(greaterThanEqual(fc, fine)))
					continue;
				int k = lqKind(texelFetch(sTD3DInputs[0], fc, 0).r);
				seen += 1;
				if (k == LQ_SOLID)
					solids += 1;
				else if (k == LQ_LIQUID)
					liquids += 1;
			}
	int kind = LQ_AIR;
	if (seen > 0 && solids * 2 >= seen)
		kind = LQ_SOLID;
	else if (liquids > 0)
		kind = LQ_LIQUID;
	TDImageStoreOutput(0, c, vec4(float(kind), 0.0, 0.0, 1.0));
}
