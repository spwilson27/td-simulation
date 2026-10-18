#include <bus3d_glsl>
#include <liquid_common_glsl>
// Liquid divergence. Each cell owns its +face. Liquid→liquid/air uses this cell's velocity,
// air→liquid uses the extrapolated velocity stored on the air cell, solids use the obstacle
// velocity, and a closed domain face is 0. in0 velocity, in1 type, in2 obstacles.
// td: output3d
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

int kindAt(ivec3 c, ivec3 res)
{
	if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, res)))
		return -1;
	return lqKind(texelFetch(sTD3DInputs[1], c, 0).r);
}

vec3 velAt(ivec3 c)
{
	return texelFetch(sTD3DInputs[0], c, 0).xyz;
}

vec3 obsAt(ivec3 c)
{
	return texelFetch(sTD3DInputs[2], c, 0).gba;
}

float flux(ivec3 owner, ivec3 res, int axis)
{
	if (any(lessThan(owner, ivec3(0))) || any(greaterThanEqual(owner, res)))
		return 0.0;
	ivec3 delta = ivec3(0);
	delta[axis] = 1;
	ivec3 nb = owner + delta;
	int okind = kindAt(owner, res);
	bool nb_in = all(greaterThanEqual(nb, ivec3(0))) && all(lessThan(nb, res));
	if (!nb_in)
	{
		if (!busSideOpen(delta))
			return 0.0;
		return okind == LQ_LIQUID ? velAt(owner)[axis] : 0.0;
	}
	int nkind = kindAt(nb, res);
	if (okind == LQ_LIQUID && (nkind == LQ_LIQUID || nkind == LQ_AIR))
		return velAt(owner)[axis];
	if (okind == LQ_LIQUID && nkind == LQ_SOLID)
		return obsAt(nb)[axis];
	if (okind == LQ_SOLID)
		return obsAt(owner)[axis];
	if (okind == LQ_AIR && nkind == LQ_LIQUID)
		return velAt(owner)[axis];
	if (okind == LQ_AIR && nkind == LQ_SOLID)
		return obsAt(nb)[axis];
	return 0.0;
}

void main()
{
	ivec3 c = busCoord();
	ivec3 res = ivec3(busRes());
	if (!busLive(c))
		return;
	if (kindAt(c, res) != LQ_LIQUID)
	{
		TDImageStoreOutput(0, c, vec4(0.0));
		return;
	}
	float div = 0.0;
	for (int axis = 0; axis < 3; axis++)
	{
		ivec3 back = ivec3(0);
		back[axis] = 1;
		div += flux(c, res, axis) - flux(c - back, res, axis);
	}
	TDImageStoreOutput(0, c, vec4(div / busCell(), 0.0, 0.0, 1.0));
}
