// Checksum of each 16x16 block. Cooking this reads every splat texel, so the later
// 1x1 readback waits for the whole sum rather than one pixel.
uniform vec4 uUnused;
layout(local_size_x = 16, local_size_y = 16) in;

void main()
{
	ivec2 outSize = ivec2(uTDOutputInfo.res.zw);
	ivec2 block = ivec2(gl_GlobalInvocationID.xy);
	if (block.x >= outSize.x || block.y >= outSize.y)
		return;
	ivec2 srcSize = textureSize(sTD2DInputs[0], 0);
	vec4 acc = vec4(0.0);
	for (int y = 0; y < 16; y++)
	{
		for (int x = 0; x < 16; x++)
		{
			ivec2 p = block * 16 + ivec2(x, y);
			if (p.x < srcSize.x && p.y < srcSize.y)
				acc += texelFetch(sTD2DInputs[0], p, 0);
		}
	}
	TDImageStoreOutput(0, ivec3(block, 0), acc);
}
