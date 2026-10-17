// Probe: compute shader with read-write output access, N passes, accumulating in place.
layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
	vec4 prev = TDImageLoadOutput(0, gl_GlobalInvocationID);
	TDImageStoreOutput(0, gl_GlobalInvocationID, prev + vec4(1.0, float(uTDPass), 0.0, 1.0));
}
