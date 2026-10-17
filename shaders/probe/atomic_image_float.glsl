// Probe: float image atomics on the output (GL_EXT_shader_atomic_float). All invocations add 1 into texel (0,0).
#extension GL_EXT_shader_atomic_float : require
layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
	imageAtomicAdd(mTDComputeOutputs[0], ivec2(0, 0), 1.0);
}
