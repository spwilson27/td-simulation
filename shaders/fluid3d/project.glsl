#include <bus3d_glsl>
// v -= grad p, with the forward difference that is the adjoint of divergence.glsl.
// A face against a solid or the domain wall contributes no pressure gradient (Neumann).
// Solids take the obstacle velocity. uLimit.x is the speed cap.
// in0 velocity, in1 pressure, in2 obstacles.
uniform vec4 uLimit;
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

bool inRange(ivec3 c, ivec3 res)
{
	return all(greaterThanEqual(c, ivec3(0))) && all(lessThan(c, res));
}

bool solidAt(ivec3 c)
{
	return busSolid(texelFetch(sTD3DInputs[2], c, 0));
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
	float h = busCell();
	vec3 grad = vec3(0.0);
	ivec3 px = c + ivec3(1, 0, 0);
	ivec3 py = c + ivec3(0, 1, 0);
	ivec3 pz = c + ivec3(0, 0, 1);
	if (!inRange(px, res))
	{
		if (busSideOpen(ivec3(1, 0, 0)))
			grad.x = (0.0 - pc) / h;
	}
	else if (!solidAt(px))
		grad.x = (texelFetch(sTD3DInputs[1], px, 0).r - pc) / h;
	if (!inRange(py, res))
	{
		if (busSideOpen(ivec3(0, 1, 0)))
			grad.y = (0.0 - pc) / h;
	}
	else if (!solidAt(py))
		grad.y = (texelFetch(sTD3DInputs[1], py, 0).r - pc) / h;
	if (!inRange(pz, res))
	{
		if (busSideOpen(ivec3(0, 0, 1)))
			grad.z = (0.0 - pc) / h;
	}
	else if (!solidAt(pz))
		grad.z = (texelFetch(sTD3DInputs[1], pz, 0).r - pc) / h;
	vec3 v = texelFetch(sTD3DInputs[0], c, 0).xyz - grad;
	float speed = length(v);
	if (speed > uLimit.x)
		v *= uLimit.x / max(speed, 1e-6);
	TDImageStoreOutput(0, c, vec4(v, 0.0));
}
