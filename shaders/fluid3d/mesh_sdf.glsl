#include <bus3d_glsl>
// Jump-flood signed-distance seeds for a triangle mesh.
// Pass 0 clears. Pass 1 (invocation 0 only) rasterises triangles into a narrow band of seeds.
// Later passes are 3D jump flooding. uInfo.x is the triangle count.
// Triangles live in a 2D texture, width 4: xyz of v0, v1, v2 and the outward normal.
// uBMin/uBMax are the mesh-local volume bounds, not the bus bounds.
uniform vec4 uInfo;
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

vec3 triVert(int row, int col)
{
	return texelFetch(sTD2DInputs[0], ivec2(col, row), 0).xyz;
}

vec3 closestTri(vec3 p, vec3 a, vec3 b, vec3 c)
{
	vec3 ab = b - a, ac = c - a, ap = p - a;
	float d1 = dot(ab, ap), d2 = dot(ac, ap);
	if (d1 <= 0.0 && d2 <= 0.0) return a;
	vec3 bp = p - b;
	float d3 = dot(ab, bp), d4 = dot(ac, bp);
	if (d3 >= 0.0 && d4 <= d3) return b;
	float vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
		return a + ab * (d1 / max(d1 - d3, 1e-8));
	vec3 cp = p - c;
	float d5 = dot(ab, cp), d6 = dot(ac, cp);
	if (d6 >= 0.0 && d5 <= d6) return c;
	float vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
		return a + ac * (d2 / max(d2 - d6, 1e-8));
	float va = d3 * d6 - d5 * d4;
	if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
		return b + (c - b) * ((d4 - d3) / max((d4 - d3) + (d5 - d6), 1e-8));
	float denom = max(va + vb + vc, 1e-8);
	return a + ab * (vb / denom) + ac * (vc / denom);
}

float packNormal(vec3 n)
{
	n = normalize(n);
	n /= max(abs(n.x) + abs(n.y) + abs(n.z), 1e-8);
	vec2 e = n.xy;
	if (n.z < 0.0)
		e = (1.0 - abs(e.yx)) * sign(e);
	e = e * 0.5 + 0.5;
	float x = clamp(floor(e.x * 2047.0 + 0.5), 0.0, 2047.0);
	float y = clamp(floor(e.y * 2047.0 + 0.5), 0.0, 2047.0);
	return x + y * 2048.0 + 1.0;
}

void raster()
{
	int count = int(uInfo.x);
	ivec3 res = ivec3(busRes());
	vec3 size = busSize();
	for (int i = 0; i < count; i++)
	{
		vec3 a = triVert(i, 0);
		vec3 b = triVert(i, 1);
		vec3 c = triVert(i, 2);
		vec3 nrm = triVert(i, 3);
		float packed = packNormal(nrm);
		vec3 mn = min(min(a, b), c);
		vec3 mx = max(max(a, b), c);
		vec3 uv0 = busWorldToUV(mn);
		vec3 uv1 = busWorldToUV(mx);
		ivec3 lo = clamp(ivec3(floor(uv0 * vec3(res))) - ivec3(1), ivec3(0), res - ivec3(1));
		ivec3 hi = clamp(ivec3(ceil(uv1 * vec3(res))) + ivec3(1), ivec3(0), res - ivec3(1));
		float band = length(size / vec3(res)) * 1.75;
		for (int z = lo.z; z <= hi.z; z++)
			for (int y = lo.y; y <= hi.y; y++)
				for (int x = lo.x; x <= hi.x; x++)
				{
					ivec3 vc = ivec3(x, y, z);
					vec3 p = busUVToWorld(busUV(vc));
					vec3 q = closestTri(p, a, b, c);
					float dist = length(p - q);
					if (dist > band)
						continue;
					vec4 prev = TDImageLoadOutput(0, vc);
					float prevD = prev.w < 0.0 ? 1e9 : length(p - prev.xyz);
					if (dist < prevD)
						TDImageStoreOutput(0, vc, vec4(q, packed));
				}
	}
}

void main()
{
	ivec3 c = busCoord();
	if (!busLive(c))
		return;
	if (uTDPass == 0)
	{
		TDImageStoreOutput(0, c, vec4(0.0, 0.0, 0.0, -1.0));
		return;
	}
	if (uTDPass == 1)
	{
		if (c == ivec3(0))
			raster();
		return;
	}
	ivec3 res = ivec3(busRes());
	int n = max(res.x, max(res.y, res.z));
	int step = max(1, n >> (uTDPass - 1));
	vec3 p = busUVToWorld(busUV(c));
	vec4 best = texelFetch(sTD3DInputs[0], c, 0);
	float bestD = best.w < 0.0 ? 1e9 : length(p - best.xyz);
	for (int z = -1; z <= 1; z++)
		for (int y = -1; y <= 1; y++)
			for (int x = -1; x <= 1; x++)
			{
				if (x == 0 && y == 0 && z == 0)
					continue;
				ivec3 nb = c + ivec3(x, y, z) * step;
				if (any(lessThan(nb, ivec3(0))) || any(greaterThanEqual(nb, res)))
					continue;
				vec4 cand = texelFetch(sTD3DInputs[0], nb, 0);
				if (cand.w < 0.0)
					continue;
				float d = length(p - cand.xyz);
				if (d < bestD)
				{
					bestD = d;
					best = cand;
				}
			}
	TDImageStoreOutput(0, c, best);
}
