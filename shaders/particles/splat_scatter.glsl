#include <splat_common_glsl>
// Pack each live particle into its spatial tile. Slot order inside a chunk follows the
// atomic return value; integer addition later does not care about that order.
// in0 deposit, in1 position, in2 per-chunk exclusive offsets, in3 per-tile ranges.
uniform vec4 uBounds;
uniform vec4 uRes;
uniform vec4 uTile;
uniform vec4 uPart;

layout(local_size_x = 16, local_size_y = 16) in;

shared int used[SPLAT_TILES];

void main()
{
	ivec2 size = max(ivec2(uPart.xy + 0.5), ivec2(1));
	if (!splatChunkLeader(ivec2(gl_WorkGroupID.xy), size.x))
		return;
	int chunkId = splatChunkId(ivec2(gl_WorkGroupID.xy), size.x);
	uint tid = gl_LocalInvocationIndex;
	if (tid == 0u)
	{
		for (int i = 0; i < SPLAT_TILES; i++)
			used[i] = 0;
	}
	barrier();

	for (int iter = 0; iter < SPLAT_CHUNK_BLOCKS; iter++)
	{
		ivec2 coord = splatParticle(chunkId, iter, int(tid), size.x);
		bool inside = coord.x < size.x && coord.y < size.y;
		ivec4 q = ivec4(0);
		int tile = 0;
		int localPix = 0;
		bool live = false;
		if (inside)
		{
			q = splatQuant(texelFetch(sTD2DInputs[0], coord, 0));
			if (splatLive(q))
			{
				vec3 pos = texelFetch(sTD2DInputs[1], coord, 0).xyz;
				ivec2 pix = splatPixel(pos, uBounds, uRes);
				ivec2 tilePx = splatTileSize(uTile);
				tile = splatTileOf(pix, tilePx);
				localPix = splatLocalOf(pix, tilePx);
				live = true;
			}
		}
		int rank = 0;
		if (live)
			rank = atomicAdd(used[tile], 1);
		// The rank is the atomic's return value, so threads do not need a barrier here.
		// A barrier inside the loop would also run on the threads that already returned.
		if (live)
		{
			ivec2 at = splatCountPix(splatChunkXY(chunkId, size.x), tile);
			int chunkOff = int(splatCountLoad(texelFetch(sTD2DInputs[2], at, 0), tile) + 0.5);
			ivec2 baseAt = ivec2(tile % SPLAT_GRID, tile / SPLAT_GRID);
			int start = int(texelFetch(sTD2DInputs[3], baseAt, 0).r + 0.5);
			int slot = start + chunkOff + rank;
			vec4 payload = vec4(float(localPix), intBitsToFloat(q.x), intBitsToFloat(q.y), intBitsToFloat(splatPackZW(q.z, q.w)));
			TDImageStoreOutput(0, ivec3(splatSlotPix(slot, size.x), 0), payload);
		}
	}
}
