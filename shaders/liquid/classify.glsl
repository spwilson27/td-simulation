#include <liquid_common_glsl>
#include <bus3d_glsl>
// Liquid where the gathered mass clears the threshold and the cell is not a real solid.
// A bus obstacle with alpha >= 0.5 is this solver's own published phi and is not a solid.
// in0 mass (w), in1 obstacles.
// td: output3d
uniform vec4 uThresh; // x mass threshold
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

void main()
{
	ivec3 c = busCoord();
	if (!busLive(c))
		return;
	float mass = texelFetch(sTD3DInputs[0], c, 0).a;
	vec4 obstacle = texelFetch(sTD3DInputs[1], c, 0);
	// Solids are the sphere / destruction field, not the bus merge (that merge contains our own phi).
	bool solid = obstacle.r < 0.0;
	int kind = LQ_AIR;
	if (solid)
		kind = LQ_SOLID;
	else if (mass >= uThresh.x)
		kind = LQ_LIQUID;
	TDImageStoreOutput(0, c, vec4(float(kind), 0.0, 0.0, 1.0));
}
