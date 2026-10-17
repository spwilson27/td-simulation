// Probe: TD atomic counter (Atomic Counters page, name "counter"). The max written value should be N-1.
layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
	uint v = atomicCounterIncrement(counter);
	TDImageStoreOutput(0, gl_GlobalInvocationID, vec4(float(v), 0.0, 0.0, 1.0));
}
