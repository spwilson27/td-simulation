#include <bus3d_glsl>
// MacCormack correction, clamped to the 8 corners of the backtrace cell.
// in0 q at the start of the step, in1 q_hat, in2 velocity, in3 obstacles.
uniform vec4 uDecay;
uniform vec4 uMac;
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main()
{
	ivec3 c = busCoord();
	if (!busLive(c))
		return;
	vec3 uv = busUV(c);
	vec4 qHat = texture(sTD3DInputs[1], uv);
	vec4 obstacle = texture(sTD3DInputs[3], uv);
	if (uMac.x <= 0.0 || busSolid(obstacle))
	{
		TDImageStoreOutput(0, c, qHat);
		return;
	}
	vec3 vel = texture(sTD3DInputs[2], uv).xyz;
	vec3 fwd = busClampUV(uv + BUS_DT * vel / busSize());
	vec4 qHatBack = texture(sTD3DInputs[1], fwd);
	vec4 decay = exp(-(uDecay + busSponge(uv) * uSponge.y) * BUS_DT);
	vec4 q = texture(sTD3DInputs[0], uv) * decay;
	vec4 corrected = qHat + 0.5 * uMac.x * (q - qHatBack);

	vec3 back = busBacktrace(sTD3DInputs[2], uv, BUS_DT);
	vec3 res = 1.0 / busTexel();
	vec3 base = (floor(back * res - 0.5) + 0.5) / res;
	vec3 t = busTexel();
	vec4 lo = vec4(1e9);
	vec4 hi = vec4(-1e9);
	for (int z = 0; z < 2; z++)
		for (int y = 0; y < 2; y++)
			for (int x = 0; x < 2; x++)
			{
				vec4 s = texture(sTD3DInputs[0], base + vec3(x, y, z) * t) * decay;
				lo = min(lo, s);
				hi = max(hi, s);
			}
	TDImageStoreOutput(0, c, clamp(corrected, lo, hi));
}
