// Shared by the 2D fixed-point splat. The GLSL TOP format menu has no r32i binding,
// so integer image atomics cannot be used. A 32x32 tile grid covers the bus. Each chunk
// stores its 1024 counts in the rgba of a 16x16 block (256 pixels x 4 channels).

#define SPLAT_GRID 32
#define SPLAT_TILES 1024
#define SPLAT_SCALE 1000000.0
#define SPLAT_CHUNK_BLOCKS 16

int splatBlocksX(int partW)
{
	return max(partW / 16, 1);
}

int splatNumChunks(ivec2 part)
{
	int blocks = splatBlocksX(part.x) * max(part.y / 16, 1);
	return max(blocks / SPLAT_CHUNK_BLOCKS, 1);
}

bool splatChunkLeader(ivec2 wg, int partW)
{
	int linear = int(wg.x) + int(wg.y) * splatBlocksX(partW);
	return linear % SPLAT_CHUNK_BLOCKS == 0;
}

int splatChunkId(ivec2 wg, int partW)
{
	int linear = int(wg.x) + int(wg.y) * splatBlocksX(partW);
	return linear / SPLAT_CHUNK_BLOCKS;
}

ivec2 splatChunkXY(int chunkId, int partW)
{
	int blocks = splatBlocksX(partW);
	return ivec2(chunkId % blocks, chunkId / blocks);
}

ivec2 splatParticle(int chunkId, int iter, int tid, int partW)
{
	int block = chunkId * SPLAT_CHUNK_BLOCKS + iter;
	int blocks = splatBlocksX(partW);
	int bx = block % blocks;
	int by = block / blocks;
	return ivec2(bx * 16 + (tid % 16), by * 16 + (tid / 16));
}

ivec2 splatPixel(vec3 pos, vec4 bounds, vec4 res)
{
	vec2 uv = clamp((pos.xy - bounds.xy) / max(bounds.zw - bounds.xy, vec2(1e-6)), vec2(0.0), vec2(1.0));
	ivec2 size = max(ivec2(res.xy + 0.5), ivec2(1));
	return clamp(ivec2(floor(uv * vec2(size))), ivec2(0), size - ivec2(1));
}

ivec2 splatTileSize(vec4 tile)
{
	return max(ivec2(tile.xy + 0.5), ivec2(1));
}

int splatTileOf(ivec2 pix, ivec2 tilePx)
{
	ivec2 t = min(pix / tilePx, ivec2(SPLAT_GRID - 1));
	return t.y * SPLAT_GRID + t.x;
}

int splatLocalOf(ivec2 pix, ivec2 tilePx)
{
	ivec2 t = min(pix / tilePx, ivec2(SPLAT_GRID - 1));
	ivec2 rel = pix - t * tilePx;
	return rel.y * tilePx.x + rel.x;
}

ivec4 splatQuant(vec4 react)
{
	float s = SPLAT_SCALE;
	return ivec4(int(round(react.x * s)), int(round(react.y * s)), int(round(react.z * s)), int(round(react.w * s)));
}

bool splatLive(ivec4 q)
{
	return any(notEqual(q, ivec4(0)));
}

int splatPackZW(int z, int w)
{
	z = clamp(z, -32768, 32767);
	w = clamp(w, -32768, 32767);
	return (z & 0xffff) | ((w & 0xffff) << 16);
}

int splatUnpack16(int packed, int shift)
{
	int v = (packed >> shift) & 0xffff;
	return (v << 16) >> 16;
}

ivec2 splatCountPix(ivec2 chunk, int tile)
{
	int local = tile % 256;
	return ivec2(chunk.x * 16 + (local % 16), chunk.y * 16 + (local / 16));
}

int splatCountChan(int tile)
{
	return tile / 256;
}

float splatCountLoad(vec4 stored, int tile)
{
	return stored[splatCountChan(tile)];
}

ivec2 splatSlotPix(int slot, int width)
{
	int w = max(width, 1);
	return ivec2(slot % w, slot / w);
}
