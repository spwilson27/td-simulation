// Deterministic particle-to-grid scatter. Float additive blending is not order-stable on this GPU,
// so two coupled bakes disagreed in the last bits. Integer adds in shared memory commute, and one
// workgroup does the whole sum. The result is a coarse grid the bus samples across the fine cells.
// in0 deposit (force xyz, heat w), in1 position.

#define BINS_X 32
#define BINS_Y 32

layout(local_size_x = 32, local_size_y = 32) in;

uniform vec4 uMode;
uniform vec4 uBounds;
uniform vec4 uBMin;
uniform vec4 uBMax;
uniform vec4 uAtlas;
uniform vec4 uAtlas2;
uniform vec4 uRes;
uniform vec4 uQuant; // x fixed-point scale

shared int bins[4 * BINS_X * BINS_Y];

ivec2 pixelFor(vec3 pos)
{
	if (uMode.x > 0.5)
	{
		vec3 vol = vec3(uAtlas.z, uAtlas.w, uAtlas2.z);
		vec3 uvw = clamp((pos - uBMin.xyz) / max(uBMax.xyz - uBMin.xyz, vec3(1e-6)), vec3(0.0), vec3(1.0));
		ivec3 ijk = ivec3(clamp(floor(uvw * vol), vec3(0.0), vol - 1.0));
		int cols = max(int(uAtlas.x), 1);
		return ivec2(ijk.z % cols * int(vol.x) + ijk.x, (ijk.z / cols) * int(vol.y) + ijk.y);
	}
	vec2 uv = clamp((pos.xy - uBounds.xy) / max(uBounds.zw - uBounds.xy, vec2(1e-6)), vec2(0.0), vec2(1.0));
	return ivec2(clamp(floor(uv * uRes.xy), vec2(0.0), uRes.xy - 1.0));
}

void main()
{
	if (gl_WorkGroupID.x != 0u || gl_WorkGroupID.y != 0u)
		return;
	uint tid = gl_LocalInvocationIndex;
	uint nBins = uint(BINS_X * BINS_Y);
	for (uint i = tid; i < 4u * nBins; i += gl_WorkGroupSize.x * gl_WorkGroupSize.y)
		bins[i] = 0;
	barrier();

	ivec2 target = ivec2(max(uRes.xy, vec2(1.0)));
	ivec2 count = textureSize(sTD2DInputs[0], 0);
	int n = count.x * count.y;
	int threads = int(gl_WorkGroupSize.x * gl_WorkGroupSize.y);
	float scale = max(uQuant.x, 1.0);
	for (int i = int(tid); i < n; i += threads)
	{
		ivec2 coord = ivec2(i % count.x, i / count.x);
		vec4 react = texelFetch(sTD2DInputs[0], coord, 0);
		if (dot(abs(react), vec4(1.0)) < 1e-20)
			continue;
		vec3 pos = texelFetch(sTD2DInputs[1], coord, 0).xyz;
		ivec2 pix = pixelFor(pos);
		int bx = clamp(pix.x * BINS_X / target.x, 0, BINS_X - 1);
		int by = clamp(pix.y * BINS_Y / target.y, 0, BINS_Y - 1);
		int cell = by * BINS_X + bx;
		atomicAdd(bins[cell], int(round(react.x * scale)));
		atomicAdd(bins[nBins + cell], int(round(react.y * scale)));
		atomicAdd(bins[2u * nBins + cell], int(round(react.z * scale)));
		atomicAdd(bins[3u * nBins + cell], int(round(react.w * scale)));
	}
	barrier();

	ivec2 outSize = ivec2(uTDOutputInfo.res.zw);
	float pixels = float(max(target.x / BINS_X, 1) * max(target.y / BINS_Y, 1));
	for (int y = int(gl_LocalInvocationID.y); y < outSize.y; y += int(gl_WorkGroupSize.y))
	{
		for (int x = int(gl_LocalInvocationID.x); x < outSize.x; x += int(gl_WorkGroupSize.x))
		{
			int bx = clamp(x * BINS_X / outSize.x, 0, BINS_X - 1);
			int by = clamp(y * BINS_Y / outSize.y, 0, BINS_Y - 1);
			int cell = by * BINS_X + bx;
			vec4 value = vec4(bins[cell], bins[nBins + cell], bins[2u * nBins + cell], bins[3u * nBins + cell]);
			TDImageStoreOutput(0, ivec3(x, y, 0), value / (scale * pixels));
		}
	}
}
