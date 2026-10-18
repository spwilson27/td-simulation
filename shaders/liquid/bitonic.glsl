#include <liquid_common_glsl>
// Power-of-two bitonic sort. Pass k reads pass k-1 through input 0. The (k, j) pair is decoded
// from uTDPass the same way flip.bitonic_passes does: stage s contributes s sub-passes.
// Keys compare cell id, then particle index, so a bin ends up in index order.
uniform vec4 uUnused;
layout(local_size_x = 8, local_size_y = 8) in;

bool keyLess(vec4 a, vec4 b)
{
	if (a.r != b.r)
		return a.r < b.r;
	return a.g < b.g;
}

void main()
{
	ivec2 size = ivec2(uTDOutputInfo.res.zw);
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	if (coord.x >= size.x || coord.y >= size.y)
		return;
	int n = size.x * size.y;
	int i = coord.x + coord.y * size.x;
	int pass = uTDPass;
	int acc = 0;
	int kexp = 1;
	int jexp = 0;
	for (int s = 1; s <= 24; s++)
	{
		if (pass < acc + s)
		{
			kexp = s;
			jexp = s - 1 - (pass - acc);
			break;
		}
		acc += s;
	}
	int k = 1 << kexp;
	int j = 1 << jexp;
	int partner = i ^ j;
	// partner stays on the texture: n is a power of two and i < n.
	partner = clamp(partner, 0, n - 1);
	ivec2 pc = lqTexel(partner, size.x);
	vec4 self = texelFetch(sTD2DInputs[0], coord, 0);
	vec4 other = texelFetch(sTD2DInputs[0], pc, 0);
	bool less = keyLess(self, other);
	bool eq = self.r == other.r && self.g == other.g;
	bool ascending = (i & k) == 0;
	bool lowGetsSelf;
	if (i < partner)
		lowGetsSelf = ascending ? (less || eq) : !(less || eq);
	else
	{
		bool aLess = keyLess(other, self);
		bool lowGetsA = ascending ? (aLess || eq) : !(aLess || eq);
		// The low index keeps its own key when lowGetsA, so this (high) index keeps its own too.
		lowGetsSelf = lowGetsA;
	}
	TDImageStoreOutput(0, ivec3(coord, 0), lowGetsSelf ? self : other);
}
