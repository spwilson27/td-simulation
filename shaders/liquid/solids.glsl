#include <bus3d_glsl>
// Nearest of the wired solid fields. Input 0 is the kinematic sphere. Further inputs (destruction
// pieces, the demo obstacles) arrive in connection order and are not the liquid phi.
// td: output3d
uniform vec4 uUnused;
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

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
