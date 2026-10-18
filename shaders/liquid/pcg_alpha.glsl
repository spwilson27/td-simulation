// alpha = (r·z) / (d·Ad). A tiny denominator stores alpha 0 so the step is a no-op.
// in0 is r·z, in1 is d·Ad, both 1x1x1. Writes (1, alpha, -alpha, 1).
// td: output3d
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{
	if (gl_GlobalInvocationID.x != 0u || gl_GlobalInvocationID.y != 0u || gl_GlobalInvocationID.z != 0u)
		return;
	float rho = texelFetch(sTD3DInputs[0], ivec3(0), 0).r;
	float denom = texelFetch(sTD3DInputs[1], ivec3(0), 0).r;
	float alpha = 0.0;
	if (abs(denom) >= 1e-12)
		alpha = rho / denom;
	TDImageStoreOutput(0, ivec3(0), vec4(1.0, alpha, -alpha, 1.0));
}
