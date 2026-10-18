#include <bus3d_glsl>
// Smoke and heat from break events. in0 is a list of bursts: xyz position, w strength (0 = unused).
uniform vec4 uDust; // x sigma, y max bursts, z heat scale, w unused

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

void main()
{
	ivec3 c = busCoord();
	if (!busLive(c))
		return;
	vec3 p = busUVToWorld(busUV(c));
	float sigma = max(uDust.x, 1e-3);
	int n = clamp(int(uDust.y + 0.5), 0, 64);
	vec4 acc = vec4(0.0);
	for (int i = 0; i < 64; i++)
	{
		if (i >= n)
			break;
		vec4 b = texelFetch(sTD2DInputs[0], ivec2(i, 0), 0);
		if (b.w <= 0.0)
			continue;
		float d2 = dot(p - b.xyz, p - b.xyz);
		float w = b.w * exp(-d2 / (2.0 * sigma * sigma));
		acc.r += w;
		acc.g += w * uDust.z;
	}
	TDImageStoreOutput(0, c, acc);
}
