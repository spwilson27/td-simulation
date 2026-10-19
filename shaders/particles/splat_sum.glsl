#include <splat_common_glsl>
// Full-resolution fixed-point sum. Each workgroup owns a 32x32 block of bus pixels and
// reduces the packed particles of that spatial tile into shared integer bins.
// in0 packed particles, in1 per-tile ranges (r start, g end).
uniform vec4 uRes;
uniform vec4 uTile;
uniform vec4 uPart;

layout(local_size_x = 32, local_size_y = 32) in;

shared int bins[4 * 32 * 32];

void main()
{
	uint tid = gl_LocalInvocationIndex;
	for (uint i = tid; i < 4u * 1024u; i += gl_WorkGroupSize.x * gl_WorkGroupSize.y)
		bins[i] = 0;
	barrier();

	ivec2 bus = max(ivec2(uRes.xy + 0.5), ivec2(1));
	ivec2 tilePx = splatTileSize(uTile);
	ivec2 origin = ivec2(gl_WorkGroupID.xy) * 32;
	ivec2 tileXY = min(origin / tilePx, ivec2(SPLAT_GRID - 1));
	int tile = tileXY.y * SPLAT_GRID + tileXY.x;
	ivec2 tileOrigin = tileXY * tilePx;
	ivec2 rel0 = origin - tileOrigin;
	ivec2 baseAt = ivec2(tile % SPLAT_GRID, tile / SPLAT_GRID);
	ivec2 range = ivec2(texelFetch(sTD2DInputs[1], baseAt, 0).rg + 0.5);
	int width = max(int(uPart.x + 0.5), 1);
	int threads = int(gl_WorkGroupSize.x * gl_WorkGroupSize.y);
	for (int slot = range.x + int(tid); slot < range.y; slot += threads)
	{
		vec4 packed = texelFetch(sTD2DInputs[0], splatSlotPix(slot, width), 0);
		int localPix = int(packed.r + 0.5);
		int lx = localPix % tilePx.x;
		int ly = localPix / tilePx.x;
		if (lx < rel0.x || ly < rel0.y || lx >= rel0.x + 32 || ly >= rel0.y + 32)
			continue;
		int cell = (ly - rel0.y) * 32 + (lx - rel0.x);
		int qx = floatBitsToInt(packed.g);
		int qy = floatBitsToInt(packed.b);
		int zw = floatBitsToInt(packed.a);
		atomicAdd(bins[cell], qx);
		atomicAdd(bins[1024 + cell], qy);
		atomicAdd(bins[2048 + cell], splatUnpack16(zw, 0));
		atomicAdd(bins[3072 + cell], splatUnpack16(zw, 16));
	}
	barrier();

	ivec2 pix = origin + ivec2(gl_LocalInvocationID.xy);
	if (pix.x >= bus.x || pix.y >= bus.y)
		return;
	int cell = int(gl_LocalInvocationID.y) * 32 + int(gl_LocalInvocationID.x);
	vec4 value = vec4(bins[cell], bins[1024 + cell], bins[2048 + cell], bins[3072 + cell]) / SPLAT_SCALE;
	TDImageStoreOutput(0, ivec3(pix, 0), value);
}
