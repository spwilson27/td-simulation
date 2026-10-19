#include <splat_common_glsl>
// Exclusive range of each spatial tile in the packed scatter buffer. r = start, g = end.
// in0 counts.
uniform vec4 uPart;

layout(local_size_x = 32, local_size_y = 32) in;

shared int totals[SPLAT_TILES];
shared int starts[SPLAT_TILES];

void main()
{
	int tile = int(gl_LocalInvocationIndex);
	ivec2 part = max(ivec2(uPart.xy + 0.5), ivec2(1));
	int chunks = splatNumChunks(part);
	int total = 0;
	if (tile < SPLAT_TILES)
	{
		for (int chunkId = 0; chunkId < chunks; chunkId++)
		{
			ivec2 chunk = splatChunkXY(chunkId, part.x);
			total += int(splatCountLoad(texelFetch(sTD2DInputs[0], splatCountPix(chunk, tile), 0), tile) + 0.5);
		}
		totals[tile] = total;
	}
	barrier();
	if (gl_LocalInvocationIndex == 0u)
	{
		int run = 0;
		for (int i = 0; i < SPLAT_TILES; i++)
		{
			starts[i] = run;
			run += totals[i];
		}
	}
	barrier();
	if (tile < SPLAT_TILES)
	{
		ivec2 at = ivec2(tile % SPLAT_GRID, tile / SPLAT_GRID);
		TDImageStoreOutput(0, ivec3(at, 0), vec4(float(starts[tile]), float(starts[tile] + totals[tile]), 0.0, 0.0));
	}
}
