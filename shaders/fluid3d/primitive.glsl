#include <bus3d_glsl>
// Analytic obstacle. uKind.x: 0 sphere, 1 box, 2 capsule. uKind.y radius. uKind.z enable. uKind.w capsule half-length (unused for sphere).
// uCenter xyz, uHalf xyz (box half extents, or capsule axis for kind 2), uVel xyz.
uniform vec4 uKind;
uniform vec4 uCenter;
uniform vec4 uHalf;
uniform vec4 uVel;
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main()
{
	ivec3 c = busCoord();
	if (!busLive(c))
		return;
	if (uKind.z < 0.5)
	{
		TDImageStoreOutput(0, c, vec4(BUS_EMPTY_DISTANCE, 0.0, 0.0, 0.0));
		return;
	}
	vec3 p = busUVToWorld(busUV(c));
	float sd;
	if (uKind.x < 0.5)
		sd = sdSphere(p, uCenter.xyz, uKind.y);
	else if (uKind.x < 1.5)
		sd = sdBox(p, uCenter.xyz, max(uHalf.xyz, vec3(1e-4)));
	else
	{
		vec3 axis = uHalf.xyz;
		float halfL = max(length(axis), 1e-4);
		axis /= halfL;
		sd = sdCapsule(p, uCenter.xyz - axis * halfL, uCenter.xyz + axis * halfL, uKind.y);
	}
	TDImageStoreOutput(0, c, vec4(sd, uVel.xyz));
}
