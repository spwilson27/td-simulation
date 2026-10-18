// Shared by the cloth compute shaders. Distance constraints and colours match python/cloth.py.

ivec2 clothTexel(int index, int width)
{
	int x = index - (index / width) * width;
	return ivec2(x, index / width);
}

int clothLinear(ivec3 c, int h)
{
	return c.x + h * (c.y + h * c.z);
}

ivec3 clothCell(vec3 p, vec3 bmin, vec3 bmax, int h)
{
	vec3 span = max(bmax - bmin, vec3(1e-6));
	ivec3 c = ivec3(floor(((p - bmin) / span) * float(h)));
	return clamp(c, ivec3(0), ivec3(h - 1));
}

vec3 clothNormal(sampler2D posTex, ivec2 c, ivec2 size)
{
	ivec2 px = clamp(c + ivec2(1, 0), ivec2(0), size - ivec2(1));
	ivec2 mx = clamp(c - ivec2(1, 0), ivec2(0), size - ivec2(1));
	ivec2 py = clamp(c + ivec2(0, 1), ivec2(0), size - ivec2(1));
	ivec2 my = clamp(c - ivec2(0, 1), ivec2(0), size - ivec2(1));
	vec3 dx = texelFetch(posTex, px, 0).xyz - texelFetch(posTex, mx, 0).xyz;
	vec3 dy = texelFetch(posTex, py, 0).xyz - texelFetch(posTex, my, 0).xyz;
	vec3 n = cross(dx, dy);
	float len = length(n);
	return len > 1e-8 ? n / len : vec3(0.0, 0.0, 1.0);
}

// Pressure follows the wind (vn * |vn| * n) and tangential drag does too.
// vRel is air velocity minus cloth velocity. cap limits that relative speed.
vec3 clothAero(vec3 n, vec3 vCloth, vec3 vAir, float area, float rho, float drag, float cap)
{
	vec3 vRel = vAir - vCloth;
	float speed = length(vRel);
	if (cap > 0.0 && speed > cap)
		vRel *= cap / speed;
	float vn = dot(vRel, n);
	vec3 vt = vRel - vn * n;
	return rho * area * vn * abs(vn) * n + drag * rho * area * length(vt) * vt;
}

// Vertex-vertex repulsion. keys hold (cell, particle index). bins hold (start, count).
// Grid neighbours within one step are skipped so the rest edges are not inflated.
// The loop cap is 48 particles per cell; python/cloth.py documents the same bound.
vec3 clothSelf(vec3 p, float inv, int id, ivec2 size, sampler2D posTex, sampler2D keys, sampler3D bins,
	vec3 bmin, vec3 bmax, int hashRes, float thickness)
{
	if (thickness <= 1e-8 || inv <= 0.0)
		return p;
	int width = size.x;
	int ix = id - (id / width) * width;
	int iy = id / width;
	ivec3 cell = clothCell(p, bmin, bmax, hashRes);
	vec3 delta = vec3(0.0);
	int count = 0;
	for (int dz = -1; dz <= 1; ++dz)
		for (int dy = -1; dy <= 1; ++dy)
			for (int dx = -1; dx <= 1; ++dx)
			{
				ivec3 c = cell + ivec3(dx, dy, dz);
				if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, ivec3(hashRes))))
					continue;
				vec4 bin = texelFetch(bins, c, 0);
				int start = int(bin.r + 0.5);
				int n = min(int(bin.g + 0.5), 48);
				for (int k = 0; k < 48; ++k)
				{
					if (k >= n)
						break;
					int other = int(texelFetch(keys, clothTexel(start + k, width), 0).g + 0.5);
					if (other == id)
						continue;
					int ox = other - (other / width) * width;
					int oy = other / width;
					if (abs(ox - ix) <= 1 && abs(oy - iy) <= 1)
						continue;
					vec4 q = texelFetch(posTex, ivec2(ox, oy), 0);
					vec3 diff = p - q.xyz;
					float dist = length(diff);
					if (dist >= thickness)
						continue;
					vec3 dir = dist > 1e-8 ? diff / dist : vec3(1.0, 0.0, 0.0);
					float wsum = inv + max(q.w, 0.0);
					if (wsum <= 0.0)
						continue;
					delta += dir * (thickness - dist) * (inv / wsum);
					count += 1;
				}
			}
	if (count > 0)
		p += delta / float(count);
	return p;
}
