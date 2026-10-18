#include <liquid_common_glsl>
#include <bus3d_glsl>
// Copy liquid velocity into a two-cell air band so an air-owned face matches the free surface.
// Pass 0 reads the forced velocity (w = 1 on liquid). Later passes read the previous band.
// in0 velocity, in1 type, in2 obstacles.
// td: output3d
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

bool knownAt(ivec3 c, ivec3 res)
{
	if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, res)))
		return false;
	if (lqKind(texelFetch(sTD3DInputs[1], c, 0).r) == LQ_LIQUID)
		return true;
	if (uTDPass > 0 && texelFetch(sTD3DInputs[0], c, 0).w > 0.5)
		return true;
	return false;
}

void main()
{
	ivec3 c = busCoord();
	ivec3 res = ivec3(busRes());
	if (!busLive(c))
		return;
	int kind = lqKind(texelFetch(sTD3DInputs[1], c, 0).r);
	if (kind == LQ_SOLID)
	{
		TDImageStoreOutput(0, c, vec4(texelFetch(sTD3DInputs[2], c, 0).gba, 0.0));
		return;
	}
	if (knownAt(c, res))
	{
		TDImageStoreOutput(0, c, vec4(texelFetch(sTD3DInputs[0], c, 0).xyz, 1.0));
		return;
	}
	vec3 acc = vec3(0.0);
	float w = 0.0;
	ivec3 dirs[6] = ivec3[6](ivec3(1, 0, 0), ivec3(-1, 0, 0), ivec3(0, 1, 0), ivec3(0, -1, 0), ivec3(0, 0, 1), ivec3(0, 0, -1));
	for (int i = 0; i < 6; i++)
	{
		ivec3 nb = c + dirs[i];
		if (!knownAt(nb, res))
			continue;
		acc += texelFetch(sTD3DInputs[0], nb, 0).xyz;
		w += 1.0;
	}
	vec3 v = w > 0.0 ? acc / w : vec3(0.0);
	TDImageStoreOutput(0, c, vec4(v, w > 0.0 ? 1.0 : 0.0));
}
