#include <bus_glsl>
// Emitter publisher: a flickering gaussian puff. The same shader writes the sources texture (uSplatVel.z = 0) or
// the forces texture (uSplatVel.z = 1); register each TOP on the bus with the matching kind.
uniform vec4 uSplatPos;     // xy world centre, z radius (world), w flicker 0..1
uniform vec4 uSplatAmount;  // per second at the centre: smoke, heat, fuel, expansion
uniform vec4 uSplatVel;     // xy acceleration (world / s^2), z output selector, w noise frequency

out vec4 fragColor;
void main()
{
	vec2 p = busUVToWorld(vUV.st);
	float d = length(p - uSplatPos.xy) / max(uSplatPos.z, 1e-5);
	float falloff = exp(-4.0 * d * d) * step(d, 1.5);
	float n = busFbm(p * uSplatVel.w + vec2(BUS_SEED, -3.0 * BUS_TIME));
	falloff *= mix(1.0, 2.0 * n, uSplatPos.w);
	vec4 outv = uSplatVel.z < 0.5 ? uSplatAmount * falloff : vec4(uSplatVel.xy * falloff, 0.0, 0.0);
	fragColor = TDOutputSwizzle(outv);
}
