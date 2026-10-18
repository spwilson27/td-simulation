#include <liquid_common_glsl>
// Exclusive prefix of one radix digit, plus the particle count of every earlier digit,
// so digit 1 starts where digit 0 ended. One workgroup of 16 threads does every tile.
// The loop is a few tens of thousands of texels; that is cheaper than a 15-pass scan
// and it stays exact in float32 (counts are below 2^24).
// uHist.x stride, uHist.y tile count. in0 is the histogram.
uniform vec4 uHist;
layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

shared uint totals[16];

void main()
{
	if (gl_WorkGroupID.x != 0u || gl_WorkGroupID.y != 0u)
		return;
	int digit = int(gl_LocalInvocationID.x);
	int stride = max(int(uHist.x + 0.5), 1);
	int tiles = max(int(uHist.y + 0.5), 1);
	uint total = 0u;
	for (int group = 0; group < tiles; group++)
	{
		ivec2 at = lqHistCoord(group, digit, tiles, stride);
		total += uint(texelFetch(sTD2DInputs[0], at, 0).r + 0.5);
	}
	totals[digit] = total;
	barrier();
	uint base = 0u;
	for (int d = 0; d < digit; d++)
		base += totals[d];
	uint run = base;
	for (int group = 0; group < tiles; group++)
	{
		ivec2 at = lqHistCoord(group, digit, tiles, stride);
		uint count = uint(texelFetch(sTD2DInputs[0], at, 0).r + 0.5);
		TDImageStoreOutput(0, ivec3(at, 0), vec4(float(run), 0.0, 0.0, 1.0));
		run += count;
	}
}
