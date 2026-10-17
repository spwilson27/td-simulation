#include <bus3d_glsl>
// Turn jump-flood seeds into a signed distance. uInfo.y is +1 or -1 (flip if the centre came out positive).
// Invalid seeds (w < 0) stay a large positive distance. uBMin/uBMax are the mesh-local volume bounds.
uniform vec4 uInfo;
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

vec3 unpackNormal(float packed)
{
	float p = packed - 1.0;
	float x = mod(p, 2048.0);
	float y = floor(p / 2048.0);
	vec2 e = vec2(x, y) / 2047.0 * 2.0 - 1.0;
	vec3 n = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
	if (n.z < 0.0)
	{
		vec2 s = sign(n.xy);
		n.xy = (1.0 - abs(n.yx)) * vec2(s.x == 0.0 ? 1.0 : s.x, s.y == 0.0 ? 1.0 : s.y);
	}
	return normalize(n);
}

void main()
{
	ivec3 c = busCoord();
	if (!busLive(c))
		return;
	vec4 seed = texelFetch(sTD3DInputs[0], c, 0);
	if (seed.w < 0.0)
	{
		TDImageStoreOutput(0, c, vec4(BUS_EMPTY_DISTANCE, 0.0, 0.0, 1.0));
		return;
	}
	vec3 p = busUVToWorld(busUV(c));
	vec3 n = unpackNormal(seed.w) * uInfo.y;
	float dist = length(p - seed.xyz);
	float sgn = dot(p - seed.xyz, n) < 0.0 ? -1.0 : 1.0;
	TDImageStoreOutput(0, c, vec4(sgn * dist, 0.0, 0.0, 1.0));
}
