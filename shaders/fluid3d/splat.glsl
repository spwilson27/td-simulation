#include <bus3d_glsl>
// Flickering gaussian emitter. uSplatVel.w < 0.5 writes sources, otherwise an acceleration (forces).
uniform vec4 uSplatPos;     // xyz centre, w radius
uniform vec4 uSplatAmount;  // smoke, heat, fuel, expansion per second at the centre
uniform vec4 uSplatVel;     // xyz acceleration, w selector (0 sources, 1 forces)
uniform vec4 uSplatNoise;   // x flicker, y frequency
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main()
{
	ivec3 c = busCoord();
	if (!busLive(c))
		return;
	vec3 p = busUVToWorld(busUV(c));
	float d = length(p - uSplatPos.xyz) / max(uSplatPos.w, 1e-5);
	float falloff = exp(-4.0 * d * d) * step(d, 1.6);
	float n = busFbm(p * uSplatNoise.y + vec3(BUS_SEED, -3.0 * BUS_TIME, 1.7));
	falloff *= mix(1.0, 2.0 * n, uSplatNoise.x);
	vec4 outv;
	if (uSplatVel.w < 0.5)
		outv = uSplatAmount * falloff;
	else
	{
		// A little sideways push so the plume billows instead of staying a laminar jet.
		float sx = busFbm(p * uSplatNoise.y + vec3(4.2, BUS_TIME, 0.3)) - 0.5;
		float sz = busFbm(p * uSplatNoise.y + vec3(BUS_TIME, 1.7, 8.1)) - 0.5;
		vec3 accel = uSplatVel.xyz + vec3(sx, 0.0, sz) * uSplatVel.y * 0.45 * uSplatNoise.x;
		outv = vec4(accel * falloff, 0.0);
	}
	TDImageStoreOutput(0, c, outv);
}
