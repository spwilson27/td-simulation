#include <liquid_common_glsl>
#include <bus3d_glsl>
// Add gravity and bus acceleration on liquid cells. Solids take the obstacle velocity.
// FLIP saves the pre-gravity velocity separately; this is the field the projection starts from.
// in0 p2g velocity, in1 type, in2 obstacles, in3 bus forces.
// td: output3d
uniform vec4 uGravity;
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

void main()
{
	ivec3 c = busCoord();
	if (!busLive(c))
		return;
	int kind = lqKind(texelFetch(sTD3DInputs[1], c, 0).r);
	vec3 v = texelFetch(sTD3DInputs[0], c, 0).xyz;
	vec4 obstacle = texelFetch(sTD3DInputs[2], c, 0);
	if (kind == LQ_SOLID)
		v = obstacle.gba;
	else if (kind == LQ_LIQUID)
	{
		vec3 force = vec3(0.0);
		if (TD_NUM_3D_INPUTS > 3)
			force = texelFetch(sTD3DInputs[3], c, 0).xyz;
		v += (uGravity.xyz + force) * BUS_DT;
	}
	else
		v = vec3(0.0);
	TDImageStoreOutput(0, c, vec4(v, kind == LQ_LIQUID ? 1.0 : 0.0));
}
