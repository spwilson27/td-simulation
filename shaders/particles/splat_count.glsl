#include <splat_common_glsl>
// One thread per particle. The workgroup's 256 shared counters are one spatial tile each.
// in0 deposit, in1 position.
uniform vec4 uBounds;
uniform vec4 uRes;
uniform vec4 uTile;
uniform vec4 uPart;

layout(local_size_x = 16, local_size_y = 16) in;

shared int counts[SPLAT_TILES];

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
			counts[i] = 0;
	}
	barrier();

	for (int iter = 0; iter < SPLAT_CHUNK_BLOCKS; iter++)
	{
		ivec2 coord = splatParticle(chunkId, iter, int(tid), size.x);
		if (coord.x < size.x && coord.y < size.y)
		{
			ivec4 q = splatQuant(texelFetch(sTD2DInputs[0], coord, 0));
			if (splatLive(q))
			{
				vec3 pos = texelFetch(sTD2DInputs[1], coord, 0).xyz;
				ivec2 pix = splatPixel(pos, uBounds, uRes);
				atomicAdd(counts[splatTileOf(pix, splatTileSize(uTile))], 1);
			}
		}
	}
	barrier();

	ivec2 chunk = splatChunkXY(chunkId, size.x);
	int local = int(tid);
	vec4 packed = vec4(float(counts[local]), float(counts[local + 256]), float(counts[local + 512]), float(counts[local + 768]));
	ivec2 at = ivec2(chunk.x * 16 + (local % 16), chunk.y * 16 + (local / 16));
	TDImageStoreOutput(0, ivec3(at, 0), packed);
}
