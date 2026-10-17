#include <bus3d_glsl>
// v -= grad p. Solids take the obstacle velocity. uLimit.x is the speed cap.
// in0 velocity, in1 pressure, in2 obstacles.
uniform vec4 uLimit;
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

float pAt(ivec3 nb, ivec3 res, float pc)
{
	if (any(lessThan(nb, ivec3(0))) || any(greaterThanEqual(nb, res)))
		return pc;
	if (busSolid(texelFetch(sTD3DInputs[2], nb, 0)))
		return pc;
	return texelFetch(sTD3DInputs[1], nb, 0).r;
}

void main()
{
	ivec3 c = busCoord();
	ivec3 res = ivec3(busRes());
	if (!busLive(c))
		return;
	vec4 obstacle = texelFetch(sTD3DInputs[2], c, 0);
	if (busSolid(obstacle))
	{
		TDImageStoreOutput(0, c, vec4(busObstacleVel(obstacle), 0.0));
		return;
	}
	float pc = texelFetch(sTD3DInputs[1], c, 0).r;
	float h2 = 2.0 * busCell();
	vec3 grad = vec3(
		pAt(c + ivec3(1, 0, 0), res, pc) - pAt(c - ivec3(1, 0, 0), res, pc),
		pAt(c + ivec3(0, 1, 0), res, pc) - pAt(c - ivec3(0, 1, 0), res, pc),
		pAt(c + ivec3(0, 0, 1), res, pc) - pAt(c - ivec3(0, 0, 1), res, pc)) / h2;
	vec3 v = texelFetch(sTD3DInputs[0], c, 0).xyz - grad;
	float speed = length(v);
	if (speed > uLimit.x)
		v *= uLimit.x / max(speed, 1e-6);
	TDImageStoreOutput(0, c, vec4(v, 0.0));
}
