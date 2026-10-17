#include <bus3d_glsl>
// Full-weighting restriction of the fine residual onto this coarser grid.
// Residual is rhs - Laplacian(p) with the FINE spacing. Weights are the tensor product of [1, 3, 3, 1] / 8,
// the adjoint of cell-centred linear prolongation, renormalised where the stencil leaves the domain so a
// constant residual stays constant. A coarse grid that did not halve falls back to the average of the fine
// cells it covers. in0 fine pressure, in1 fine rhs, in2 obstacles (sampled in UV).
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

bool solidFine(ivec3 fc, ivec3 fine)
{
	if (any(lessThan(fc, ivec3(0))) || any(greaterThanEqual(fc, fine)))
		return true;
	vec3 uv = (vec3(fc) + 0.5) / vec3(fine);
	return texture(sTD3DInputs[2], clamp(uv, vec3(0.0), vec3(1.0))).r < 0.0;
}

float pFine(ivec3 nb, ivec3 fine, float pc, ivec3 dir)
{
	if (any(lessThan(nb, ivec3(0))) || any(greaterThanEqual(nb, fine)))
		return busPressureGhost(dir, pc);
	if (solidFine(nb, fine))
		return pc;
	return texelFetch(sTD3DInputs[0], nb, 0).r;
}

float residualAt(ivec3 fc, ivec3 fine, float h)
{
	if (solidFine(fc, fine))
		return 0.0;
	float pc = texelFetch(sTD3DInputs[0], fc, 0).r;
	float sum = pFine(fc + ivec3(1, 0, 0), fine, pc, ivec3(1, 0, 0)) + pFine(fc - ivec3(1, 0, 0), fine, pc, ivec3(-1, 0, 0))
	          + pFine(fc + ivec3(0, 1, 0), fine, pc, ivec3(0, 1, 0)) + pFine(fc - ivec3(0, 1, 0), fine, pc, ivec3(0, -1, 0))
	          + pFine(fc + ivec3(0, 0, 1), fine, pc, ivec3(0, 0, 1)) + pFine(fc - ivec3(0, 0, 1), fine, pc, ivec3(0, 0, -1));
	float rhs = texelFetch(sTD3DInputs[1], fc, 0).r;
	return rhs - (sum - 6.0 * pc) / (h * h);
}

float w1(int d)
{
	if (d == -1 || d == 2)
		return 0.125;
	if (d == 0 || d == 1)
		return 0.375;
	return 0.0;
}

void main()
{
	ivec3 c = busCoord();
	ivec3 coarse = ivec3(busRes());
	if (!busLive(c))
		return;
	ivec3 fine = textureSize(sTD3DInputs[0], 0);
	float h = busSize().x / float(fine.x);
	bool halved = fine == coarse * 2;
	float acc = 0.0;
	float wsum = 0.0;
	if (halved)
	{
		for (int dz = -1; dz <= 2; dz++)
			for (int dy = -1; dy <= 2; dy++)
				for (int dx = -1; dx <= 2; dx++)
				{
					ivec3 fc = c * 2 + ivec3(dx, dy, dz);
					if (any(lessThan(fc, ivec3(0))) || any(greaterThanEqual(fc, fine)))
						continue;
					if (solidFine(fc, fine))
						continue;
					float w = w1(dx) * w1(dy) * w1(dz);
					acc += w * residualAt(fc, fine, h);
					wsum += w;
				}
	}
	else
	{
		ivec3 base = c * fine / coarse;
		ivec3 endp = max((c + ivec3(1)) * fine / coarse, base + ivec3(1));
		for (int z = base.z; z < endp.z; z++)
			for (int y = base.y; y < endp.y; y++)
				for (int x = base.x; x < endp.x; x++)
				{
					ivec3 fc = ivec3(x, y, z);
					if (solidFine(fc, fine))
						continue;
					acc += residualAt(fc, fine, h);
					wsum += 1.0;
				}
	}
	float r = wsum > 0.0 ? acc / wsum : 0.0;
	if (solidFine(ivec3((vec3(c) + 0.5) * vec3(fine) / vec3(coarse)), fine))
		r = 0.0;
	TDImageStoreOutput(0, c, vec4(r, 0.0, 0.0, 1.0));
}
