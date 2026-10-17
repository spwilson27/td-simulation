// Probe: workgroup shared-memory integer atomics. Every texel of a workgroup should read 64.
layout (local_size_x = 8, local_size_y = 8) in;
shared uint sCount;
void main()
{
	if (gl_LocalInvocationIndex == 0u) sCount = 0u;
	barrier();
	atomicAdd(sCount, 1u);
	barrier();
	TDImageStoreOutput(0, gl_GlobalInvocationID, vec4(float(sCount), 0.0, 0.0, 1.0));
}
