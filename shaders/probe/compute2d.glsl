// Probe: compute shader writes 2D float32 values (precision, negatives, values > 1).
layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
	vec2 p = vec2(gl_GlobalInvocationID.xy);
	TDImageStoreOutput(0, gl_GlobalInvocationID, vec4(p.x, p.y, p.x * 1000.5, -p.y - 0.25));
}
