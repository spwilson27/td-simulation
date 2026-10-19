// One texel. numpyArray of this TOP is the GPU fence for the sum.
uniform vec4 uUnused;
layout(local_size_x = 1, local_size_y = 1) in;

void main()
{
	if (gl_GlobalInvocationID.x != 0u || gl_GlobalInvocationID.y != 0u)
		return;
	ivec2 size = textureSize(sTD2DInputs[0], 0);
	vec4 acc = vec4(0.0);
	for (int y = 0; y < size.y; y++)
		for (int x = 0; x < size.x; x++)
			acc += texelFetch(sTD2DInputs[0], ivec2(x, y), 0);
	TDImageStoreOutput(0, ivec3(0, 0, 0), acc);
}
