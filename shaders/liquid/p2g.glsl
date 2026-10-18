#include <liquid_common_glsl>
// Gather mass and momentum. Each cell walks its 3^3 neighbour bins in particle-index order
// (the sort put them there), so the sum does not depend on which thread runs first.
// in0 sorted keys, in1 bins, in2 positions, in3 velocities (via tops).
// xyz momentum, w mass.
uniform vec4 uBMin;
uniform vec4 uBMax;
uniform vec4 uPart; // xy particle texture size
// td: output3d
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

void main()
{
	ivec3 c = ivec3(gl_GlobalInvocationID.xyz);
	ivec3 res = ivec3(uTDOutputInfo.res.zw, uTDOutputInfo.depth.y);
	if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, res)))
		return;
	vec3 size = uBMax.xyz - uBMin.xyz;
	vec3 h = size / vec3(res);
	vec3 centre = uBMin.xyz + (vec3(c) + 0.5) * h;
	int width = int(uPart.x + 0.5);
	vec3 mom = vec3(0.0);
	float mass = 0.0;
	for (int dz = -1; dz <= 1; dz++)
		for (int dy = -1; dy <= 1; dy++)
			for (int dx = -1; dx <= 1; dx++)
			{
				ivec3 nc = c + ivec3(dx, dy, dz);
				if (any(lessThan(nc, ivec3(0))) || any(greaterThanEqual(nc, res)))
					continue;
				vec2 bin = texelFetch(sTD3DInputs[0], nc, 0).rg;
				int start = int(bin.r + 0.5);
				int cnt = min(int(bin.g + 0.5), 64);
				for (int n = 0; n < 64; n++)
				{
					if (n >= cnt)
						break;
					vec4 key = texelFetch(sTD2DInputs[0], lqTexel(start + n, width), 0);
					if (key.b < 0.5)
						continue;
					ivec2 pc = lqTexel(int(key.g + 0.5), width);
					vec4 pp = texelFetch(sTD2DInputs[1], pc, 0);
					if (pp.a <= 0.0)
						continue;
					float w = lqWeight((pp.xyz - centre) / h);
					vec3 pv = texelFetch(sTD2DInputs[2], pc, 0).xyz;
					mass += w * pp.a;
					mom += w * pp.a * pv;
				}
			}
	TDImageStoreOutput(0, c, vec4(mom, mass));
}
