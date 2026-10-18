#include <bus3d_glsl>
#include <liquid_common_glsl>
// v -= grad p, the adjoint of divergence.glsl. Solids are overwritten with the obstacle velocity.
// in0 velocity, in1 pressure, in2 type, in3 obstacles.
// td: output3d
uniform vec4 uLimit;
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

int kindAt(ivec3 c, ivec3 res)
{
	if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, res)))
		return -1;
	return lqKind(texelFetch(sTD3DInputs[2], c, 0).r);
}

float pressureAt(ivec3 c)
{
	return texelFetch(sTD3DInputs[1], c, 0).r;
}

void main()
{
	ivec3 c = busCoord();
	ivec3 res = ivec3(busRes());
	if (!busLive(c))
		return;
	int kind = kindAt(c, res);
	vec4 obstacle = texelFetch(sTD3DInputs[3], c, 0);
	if (kind == LQ_SOLID)
	{
		TDImageStoreOutput(0, c, vec4(obstacle.gba, 0.0));
		return;
	}
	float pc = pressureAt(c);
	float h = busCell();
	vec3 grad = vec3(0.0);
	ivec3 delta;
	for (int axis = 0; axis < 3; axis++)
	{
		delta = ivec3(0);
		delta[axis] = 1;
		ivec3 nb = c + delta;
		bool nb_in = all(greaterThanEqual(nb, ivec3(0))) && all(lessThan(nb, res));
		int nkind = nb_in ? kindAt(nb, res) : -1;
		float g = 0.0;
		if (!nb_in)
		{
			if (busSideOpen(delta) && kind == LQ_LIQUID)
				g = (0.0 - pc) / h;
		}
		else if (kind == LQ_LIQUID && nkind == LQ_LIQUID)
			g = (pressureAt(nb) - pc) / h;
		else if (kind == LQ_LIQUID && nkind == LQ_AIR)
			g = (0.0 - pc) / h;
		else if (kind == LQ_AIR && nkind == LQ_LIQUID)
			g = pressureAt(nb) / h;
		grad[axis] = g;
	}
	vec3 v = texelFetch(sTD3DInputs[0], c, 0).xyz - grad;
	float speed = length(v);
	float cap = max(uLimit.x, 0.0);
	if (cap > 0.0 && speed > cap)
		v *= cap / max(speed, 1e-6);
	TDImageStoreOutput(0, c, vec4(v, 0.0));
}
