// beta = rho / rho_prev, and 0 on the first iteration or a zero denominator.
// in0 is r·z (1x1x1), in1 is the previous r·z. Writes (1, beta, rho, 1).
// td: output3d
uniform vec4 uFirst;
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{
	if (gl_GlobalInvocationID.x != 0u || gl_GlobalInvocationID.y != 0u || gl_GlobalInvocationID.z != 0u)
		return;
	float rho = texelFetch(sTD3DInputs[0], ivec3(0), 0).r;
	float prev = texelFetch(sTD3DInputs[1], ivec3(0), 0).r;
	float beta = 0.0;
	if (uFirst.x < 0.5 && abs(prev) > 1e-18)
		beta = rho / prev;
	TDImageStoreOutput(0, ivec3(0), vec4(1.0, beta, rho, 1.0));
}
