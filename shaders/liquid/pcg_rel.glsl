// ||r|| / ||b|| from the two 1x1x1 sums r·r and b·b. One read of this ends the step.
// td: output3d
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{
	if (gl_GlobalInvocationID.x != 0u || gl_GlobalInvocationID.y != 0u || gl_GlobalInvocationID.z != 0u)
		return;
	float rr = texelFetch(sTD3DInputs[0], ivec3(0), 0).r;
	float bb = max(texelFetch(sTD3DInputs[1], ivec3(0), 0).r, 1e-30);
	TDImageStoreOutput(0, ivec3(0), vec4(sqrt(max(rr, 0.0) / bb), 0.0, 0.0, 1.0));
}
