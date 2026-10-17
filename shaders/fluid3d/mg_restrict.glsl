#include <bus3d_glsl>
// Average the fine residual onto this coarser grid. Residual is rhs - Laplacian(p), with the fine grid's h.
// in0 fine pressure, in1 fine rhs, in2 obstacles (sampled in UV, so any resolution works).
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

bool solidAt(ivec3 nb, ivec3 res)
{
	vec3 uv = clamp((vec3(nb) + 0.5) / vec3(res), vec3(0.0), vec3(1.0));
	return texture(sTD3DInputs[2], uv).r < 0.0;
}

float pFine(ivec3 nb, ivec3 res, float pc)
{
	if (any(lessThan(nb, ivec3(0))) || any(greaterThanEqual(nb, res)))
		return pc;
	if (solidAt(nb, res))
		return pc;
	return texelFetch(sTD3DInputs[0], nb, 0).r;
}

void main()
{
	ivec3 c = busCoord();
	ivec3 coarse = ivec3(busRes());
	if (!busLive(c))
		return;
	ivec3 fine = textureSize(sTD3DInputs[0], 0);
	float h = busSize().x / float(fine.x);
	ivec3 base = c * fine / coarse;
	ivec3 endp = max((c + ivec3(1)) * fine / coarse, base + ivec3(1));
	float acc = 0.0;
	float n = 0.0;
	for (int z = base.z; z < endp.z; z++)
		for (int y = base.y; y < endp.y; y++)
			for (int x = base.x; x < endp.x; x++)
			{
				ivec3 fc = ivec3(x, y, z);
				float pc = texelFetch(sTD3DInputs[0], fc, 0).r;
				float sum = pFine(fc + ivec3(1, 0, 0), fine, pc) + pFine(fc - ivec3(1, 0, 0), fine, pc)
				          + pFine(fc + ivec3(0, 1, 0), fine, pc) + pFine(fc - ivec3(0, 1, 0), fine, pc)
				          + pFine(fc + ivec3(0, 0, 1), fine, pc) + pFine(fc - ivec3(0, 0, 1), fine, pc);
				float rhs = texelFetch(sTD3DInputs[1], fc, 0).r;
				acc += rhs - (sum - 6.0 * pc) / (h * h);
				n += 1.0;
			}
	TDImageStoreOutput(0, c, vec4(acc / max(n, 1.0), 0.0, 0.0, 1.0));
}
