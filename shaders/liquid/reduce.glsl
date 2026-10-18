#include <bus3d_glsl>
// Sum a block of the input into each output cell. The last pass is 1x1x1 and sums whatever remains.
// in0 scalar field.
// td: output3d
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

void main()
{
	ivec3 c = busCoord();
	ivec3 outRes = ivec3(busRes());
	if (!busLive(c))
		return;
	ivec3 inRes = textureSize(sTD3DInputs[0], 0);
	// Every stage quarters the grid, so a block is 4 except the last step off a non-multiple, which is ≤ 4.
	ivec3 block = max(inRes / max(outRes, ivec3(1)), ivec3(1));
	float acc = 0.0;
	for (int z = 0; z < 4; z++)
		for (int y = 0; y < 4; y++)
			for (int x = 0; x < 4; x++)
			{
				if (x >= block.x || y >= block.y || z >= block.z)
					continue;
				ivec3 fc = c * block + ivec3(x, y, z);
				if (any(greaterThanEqual(fc, inRes)))
					continue;
				acc += texelFetch(sTD3DInputs[0], fc, 0).r;
			}
	TDImageStoreOutput(0, c, vec4(acc, 0.0, 0.0, 1.0));
}
