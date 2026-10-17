// Bus merge: sum of every 3D input. Input 0 is a zero volume that fixes resolution and format.
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main()
{
	ivec3 c = ivec3(gl_GlobalInvocationID.xyz);
	vec4 acc = vec4(0.0);
	for (int i = 0; i < TD_NUM_3D_INPUTS; i++)
		acc += texelFetch(sTD3DInputs[i], c, 0);
	TDImageStoreOutput(0, c, acc);
}
