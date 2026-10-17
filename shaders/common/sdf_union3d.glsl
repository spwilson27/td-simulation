// Bus merge for 3D obstacles: keep the nearest signed distance and that obstacle's velocity.
// Input 0 is the empty field (distance BUS_EMPTY_DISTANCE).
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main()
{
	ivec3 c = ivec3(gl_GlobalInvocationID.xyz);
	vec4 best = texelFetch(sTD3DInputs[0], c, 0);
	for (int i = 1; i < TD_NUM_3D_INPUTS; i++)
	{
		vec4 o = texelFetch(sTD3DInputs[i], c, 0);
		if (o.r < best.r)
			best = o;
	}
	TDImageStoreOutput(0, c, best);
}
