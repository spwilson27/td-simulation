#include <cloth_common_glsl>
// One thread per hash cell. Binary-search the sorted keys for this cell's run.
// in0 sorted keys. r = start, g = count.
uniform vec4 uPart;
// td: output3d
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

int lowerBound(int width, int n, float cell)
{
	int lo = 0;
	int hi = n;
	for (int iter = 0; iter < 24; ++iter)
	{
		if (lo >= hi)
			break;
		int mid = (lo + hi) >> 1;
		float c = texelFetch(sTD2DInputs[0], clothTexel(mid, width), 0).r;
		if (c < cell)
			lo = mid + 1;
		else
			hi = mid;
	}
	return lo;
}

void main()
{
	ivec3 c = ivec3(gl_GlobalInvocationID.xyz);
	ivec3 res = ivec3(int(uTDOutputInfo.res.z), int(uTDOutputInfo.res.w), int(uTDOutputInfo.depth.y + 0.5));
	if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, res)))
		return;
	int width = int(uPart.x + 0.5);
	int n = int(uPart.z + 0.5);
	int h = res.x;
	float cell = float(clothLinear(c, h));
	int start = lowerBound(width, n, cell);
	int endp = lowerBound(width, n, cell + 1.0);
	TDImageStoreOutput(0, c, vec4(float(start), float(max(endp - start, 0)), 0.0, 1.0));
}
