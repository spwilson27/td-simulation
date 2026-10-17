#include <bus_glsl>
// MacCormack correction of a semi-Lagrangian result: advect q_hat backwards, halve the round-trip error and add it
// back, clamped to the values around the backtrace point so the correction cannot create new extrema.
// in0: q (start of step), in1: q_hat (advect.glsl output), in2: velocity, in3: obstacles
uniform vec4 uDecay;      // same decay as the advect pass
uniform vec4 uMac;        // x: correction strength 0..1 (0 returns q_hat unchanged)

out vec4 fragColor;
void main()
{
	vec2 uv = vUV.st;
	vec4 qHat = texture(sTD2DInputs[1], uv);
	vec4 obstacle = texture(sTD2DInputs[3], uv);
	if (uMac.x <= 0.0 || busSolid(obstacle))
	{
		fragColor = TDOutputSwizzle(qHat);
		return;
	}
	vec2 vel = texture(sTD2DInputs[2], uv).xy;
	vec2 fwd = busClampUV(uv + BUS_DT * vel / busSize());
	vec4 qHatBack = texture(sTD2DInputs[1], fwd);
	vec4 decay = exp(-(uDecay + busSponge(uv) * uSponge.y) * BUS_DT);
	vec4 q = texture(sTD2DInputs[0], uv) * decay;
	vec4 corrected = qHat + 0.5 * uMac.x * (q - qHatBack);

	vec2 back = busBacktrace(sTD2DInputs[2], uv, BUS_DT);
	vec2 res = 1.0 / busTexel();
	vec2 base = (floor(back * res - 0.5) + 0.5) / res;
	vec4 q00 = texture(sTD2DInputs[0], base);
	vec4 q10 = texture(sTD2DInputs[0], base + vec2(busTexel().x, 0.0));
	vec4 q01 = texture(sTD2DInputs[0], base + vec2(0.0, busTexel().y));
	vec4 q11 = texture(sTD2DInputs[0], base + busTexel());
	vec4 lo = min(min(q00, q10), min(q01, q11)) * decay;
	vec4 hi = max(max(q00, q10), max(q01, q11)) * decay;
	fragColor = TDOutputSwizzle(clamp(corrected, lo, hi));
}
