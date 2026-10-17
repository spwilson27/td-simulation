// Probe: compute shader writes every voxel of a 3D texture.
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main()
{
	vec3 p = vec3(gl_GlobalInvocationID.xyz);
	TDImageStoreOutput(0, gl_GlobalInvocationID, vec4(p, 1.0));
}
