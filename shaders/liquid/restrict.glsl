#include <bus3d_glsl>
#include <liquid_common_glsl>
// Full-weighting restriction of the fine residual. Weights are NOT renormalised: doing that
// treats missing air samples as copies of the surface residual and the V-cycle diverges above 16^3.
// in0 fine pressure, in1 fine rhs, in2 fine type.
// td: output3d
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

int kindFine(ivec3 fc, ivec3 fine)
{
	if (any(lessThan(fc, ivec3(0))) || any(greaterThanEqual(fc, fine)))
		return -1;
	vec3 uv = (vec3(fc) + 0.5) / vec3(fine);
	return lqKind(texture(sTD3DInputs[2], clamp(uv, vec3(0.0), vec3(1.0))).r);
}

float pFine(ivec3 nb, ivec3 fine, float pc, ivec3 dir)
{
	if (any(lessThan(nb, ivec3(0))) || any(greaterThanEqual(nb, fine)))
		return busPressureGhost(dir, pc);
	int k = kindFine(nb, fine);
	if (k == LQ_LIQUID)
		return texelFetch(sTD3DInputs[0], nb, 0).r;
	if (k == LQ_AIR)
		return 0.0;
	return pc;
}

float residualAt(ivec3 fc, ivec3 fine, float h)
{
	if (kindFine(fc, fine) != LQ_LIQUID)
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
					if (kindFine(fc, fine) != LQ_LIQUID)
						continue;
					float w = w1(dx) * w1(dy) * w1(dz);
					acc += w * residualAt(fc, fine, h);
					wsum += w;
				}
		// Dividing by the partial liquid weight amplifies the free-surface residual and the
		// V-cycle diverges above 16^3. Leaving the weights un-normalised keeps the interior
		// (where the stencil is complete and wsum is 1) exact and stays stable.
	}
	else
	{
		ivec3 base = c * fine / coarse;
		ivec3 endp = max((c + ivec3(1)) * fine / coarse, base + ivec3(1));
		for (int z = 0; z < 4; z++)
			for (int y = 0; y < 4; y++)
				for (int x = 0; x < 4; x++)
				{
					ivec3 fc = base + ivec3(x, y, z);
					if (any(greaterThanEqual(fc, endp)) || any(greaterThanEqual(fc, fine)))
						continue;
					if (kindFine(fc, fine) != LQ_LIQUID)
						continue;
					acc += residualAt(fc, fine, h);
					wsum += 1.0;
				}
		if (wsum > 0.0)
			acc /= wsum;
	}
	ivec3 probe = ivec3((vec3(c) + 0.5) * vec3(fine) / vec3(coarse));
	if (kindFine(probe, fine) != LQ_LIQUID)
		acc = 0.0;
	TDImageStoreOutput(0, c, vec4(acc, 0.0, 0.0, 1.0));
}
