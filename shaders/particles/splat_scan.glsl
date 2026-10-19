#include <splat_common_glsl>
// Exclusive prefix of each spatial tile across particle chunks. Workgroup 0 writes the
// whole buffer; every other workgroup exists only because the output is particle-sized.
// in0 counts.
uniform vec4 uPart;

layout(local_size_x = 16, local_size_y = 16) in;

void main()
{
	if (gl_WorkGroupID.x != 0u || gl_WorkGroupID.y != 0u)
		return;
	// One thread owns one 16x16-block pixel, which holds four tile counts.
	int local = int(gl_LocalInvocationIndex);
	if (local >= 256)
		return;
	ivec2 part = max(ivec2(uPart.xy + 0.5), ivec2(1));
	int chunks = splatNumChunks(part);
	ivec4 running = ivec4(0);
	for (int chunkId = 0; chunkId < chunks; chunkId++)
	{
		ivec2 chunk = splatChunkXY(chunkId, part.x);
		ivec2 at = ivec2(chunk.x * 16 + (local % 16), chunk.y * 16 + (local / 16));
		ivec4 count = ivec4(texelFetch(sTD2DInputs[0], at, 0) + 0.5);
		TDImageStoreOutput(0, ivec3(at, 0), vec4(running));
		running += count;
	}
}
