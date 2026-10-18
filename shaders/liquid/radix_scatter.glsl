#include <liquid_common_glsl>
// Stable scatter of one radix digit. Workgroups are 64 consecutive particle indices,
// matching the histogram tiles. Rank is the count of the same digit in earlier lanes
// (texelFetch, not shared memory: a shared array in this shader stored nothing).
// in0 keys, in1 exclusive tile offsets (tiles x 16). uShift.x is the bit offset.
uniform vec4 uShift;
uniform vec4 uHist;
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main()
{
	ivec2 size = ivec2(uTDOutputInfo.res.zw);
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	if (coord.x >= size.x || coord.y >= size.y)
		return;
	int width = size.x;
	int id = coord.x + coord.y * width;
	vec4 key = texelFetch(sTD2DInputs[0], coord, 0);
	uint shift = uint(uShift.x + 0.5);
	uint digit = (uint(key.r + 0.5) >> shift) & 15u;
	uint lane = gl_LocalInvocationIndex;
	int begin = id - int(lane);
	uint rank = 0u;
	for (uint i = 0u; i < lane; i++)
	{
		float cell = texelFetch(sTD2DInputs[0], lqTexel(begin + int(i), width), 0).r;
		uint other = (uint(cell + 0.5) >> shift) & 15u;
		if (other == digit)
			rank++;
	}
	int groupsX = max(width / 64, 1);
	int group = int(gl_WorkGroupID.x) + int(gl_WorkGroupID.y) * groupsX;
	ivec2 at = lqHistCoord(group, int(digit), int(uHist.y + 0.5), int(uHist.x + 0.5));
	uint base = uint(texelFetch(sTD2DInputs[1], at, 0).r + 0.5);
	ivec2 dst = lqTexel(int(base + rank), width);
	TDImageStoreOutput(0, ivec3(dst, 0), key);
}
