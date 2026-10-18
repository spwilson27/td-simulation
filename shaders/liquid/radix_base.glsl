#include <liquid_common_glsl>
// Add the exclusive prefix of earlier digit-rows onto a per-row exclusive scan.
// in0 scan: r is exclusive within the digit, g is that tile's original count.
// The last tile's r+g is the digit's total. Sums stay inside the float32 integer range
// (particle counts are below 2^24).
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main()
{
	ivec2 size = ivec2(uTDOutputInfo.res.zw);
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	if (coord.x >= size.x || coord.y >= size.y)
		return;
	vec4 mine = texelFetch(sTD2DInputs[0], coord, 0);
	float add = 0.0;
	int last = size.x - 1;
	for (int d = 0; d < coord.y; d++)
	{
		vec4 tail = texelFetch(sTD2DInputs[0], ivec2(last, d), 0);
		add += tail.r + tail.g;
	}
	TDImageStoreOutput(0, ivec3(coord, 0), vec4(mine.r + add, 0.0, 0.0, 1.0));
}
