#include <liquid_common_glsl>
// One thread per histogram texel. A tile is 64 consecutive particle indices.
// uHist.x is the texture stride, uHist.y the tile count. uShift.x is the bit offset.
uniform vec4 uHist;
uniform vec4 uShift;
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main()
{
	ivec2 size = ivec2(uTDOutputInfo.res.zw);
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	if (coord.x >= size.x || coord.y >= size.y)
		return;
	int stride = max(int(uHist.x + 0.5), 1);
	int tiles = max(int(uHist.y + 0.5), 1);
	int rows = max((tiles + stride - 1) / stride, 1);
	if (coord.y / rows >= 16)
		return;
	int digit = coord.y / rows;
	int row = coord.y - digit * rows;
	int group = coord.x + row * stride;
	if (group >= tiles)
		return;
	ivec2 keySize = textureSize(sTD2DInputs[0], 0);
	int width = keySize.x;
	int n = keySize.x * keySize.y;
	int begin = group * 64;
	uint shift = uint(uShift.x + 0.5);
	int count = 0;
	int endp = min(begin + 64, n);
	for (int id = begin; id < endp; id++)
	{
		float cell = texelFetch(sTD2DInputs[0], lqTexel(id, width), 0).r;
		uint bits = uint(cell + 0.5);
		if (int((bits >> shift) & 15u) == digit)
			count++;
	}
	TDImageStoreOutput(0, ivec3(coord, 0), vec4(float(count), 0.0, 0.0, 1.0));
}
