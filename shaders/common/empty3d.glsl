#include <bus3d_glsl>
// Obstacle field with nothing in it. Input 0 of the 3D obstacle merge.
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main()
{
	ivec3 c = ivec3(gl_GlobalInvocationID.xyz);
	if (!busLive(c))
		return;
	TDImageStoreOutput(0, c, vec4(BUS_EMPTY_DISTANCE, 0.0, 0.0, 0.0));
}
