#include <bus3d_glsl>
// Central-difference divergence minus combustion expansion.
// in0 velocity, in1 obstacles, in2 scalars (a = expansion).
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

vec3 velAt(ivec3 nb, ivec3 res)
{
	if (any(lessThan(nb, ivec3(0))) || any(greaterThanEqual(nb, res)))
		return vec3(0.0);
	vec4 o = texelFetch(sTD3DInputs[1], nb, 0);
	return busSolid(o) ? busObstacleVel(o) : texelFetch(sTD3DInputs[0], nb, 0).xyz;
}

void main()
{
	ivec3 c = busCoord();
	ivec3 res = ivec3(busRes());
	if (!busLive(c))
		return;
	float h2 = 2.0 * busCell();
	vec3 vr = velAt(c + ivec3(1, 0, 0), res);
	vec3 vl = velAt(c - ivec3(1, 0, 0), res);
	vec3 vu = velAt(c + ivec3(0, 1, 0), res);
	vec3 vd = velAt(c - ivec3(0, 1, 0), res);
	vec3 vf = velAt(c + ivec3(0, 0, 1), res);
	vec3 vb = velAt(c - ivec3(0, 0, 1), res);
	float div = (vr.x - vl.x + vu.y - vd.y + vf.z - vb.z) / h2;
	div -= texelFetch(sTD3DInputs[2], c, 0).a;
	TDImageStoreOutput(0, c, vec4(div, 0.0, 0.0, 1.0));
}
