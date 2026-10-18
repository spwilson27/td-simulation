#include <liquid_common_glsl>
// G2P. uFlip.x is the FLIP blend (0 PIC, 1 FLIP). The saved grid is the pre-gravity P2G velocity,
// so the blend is v = (1-a) pic + a (v_particle + pic - p2g).
// in0 pos, in1 particle vel, 3D in0 projected velocity, 3D in1 p2g velocity.
uniform vec4 uBMin;
uniform vec4 uBMax;
uniform vec4 uGrid;
uniform vec4 uFlip;
layout(local_size_x = 8, local_size_y = 8) in;

vec3 gather(sampler3D field, vec3 pos, vec3 h, ivec3 res)
{
	vec3 pic = vec3(0.0);
	ivec3 base = ivec3(floor((pos - uBMin.xyz) / h));
	for (int dz = -1; dz <= 1; dz++)
		for (int dy = -1; dy <= 1; dy++)
			for (int dx = -1; dx <= 1; dx++)
			{
				ivec3 cell = base + ivec3(dx, dy, dz);
				if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, res)))
					continue;
				vec3 centre = uBMin.xyz + (vec3(cell) + 0.5) * h;
				float w = lqWeight((pos - centre) / h);
				pic += w * texelFetch(field, cell, 0).xyz;
			}
	return pic;
}

void main()
{
	ivec2 size = ivec2(uTDOutputInfo.res.zw);
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	if (coord.x >= size.x || coord.y >= size.y)
		return;
	vec4 pos = texelFetch(sTD2DInputs[0], coord, 0);
	vec4 vel = texelFetch(sTD2DInputs[1], coord, 0);
	if (pos.a <= 0.0)
	{
		TDImageStoreOutput(0, ivec3(coord, 0), vec4(0.0));
		return;
	}
	ivec3 res = max(ivec3(uGrid.xyz + 0.5), ivec3(1));
	vec3 h = (uBMax.xyz - uBMin.xyz) / vec3(res);
	vec3 pic = gather(sTD3DInputs[0], pos.xyz, h, res);
	vec3 prev = gather(sTD3DInputs[1], pos.xyz, h, res);
	float blend = clamp(uFlip.x, 0.0, 1.0);
	vec3 outv = (1.0 - blend) * pic + blend * (vel.xyz + pic - prev);
	float cap = uFlip.y;
	float speed = length(outv);
	if (cap > 0.0 && speed > cap)
		outv *= cap / speed;
	TDImageStoreOutput(0, ivec3(coord, 0), vec4(outv, 1.0));
}
