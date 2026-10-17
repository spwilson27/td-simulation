#include <bus_glsl>
// Body forces on the velocity field: buoyancy (hot rises, smoke sinks), bus forces, wind drag and vorticity
// confinement (re-injects the small swirls numerical diffusion removes).
// in0: velocity (advected), in1: scalars (after combustion), in2: forces, in3: obstacles
uniform vec4 uForce;      // x buoyancy per unit temperature, y weight per unit smoke, z vorticity confinement
uniform vec4 uWind;       // xy wind velocity, z drag rate towards the wind (1/s)

vec2 velAt(vec2 uv) { return texture(sTD2DInputs[0], busClampUV(uv)).xy; }

float curlAt(vec2 uv)
{
	vec2 t = busTexel();
	float dvy = velAt(uv + vec2(t.x, 0)).y - velAt(uv - vec2(t.x, 0)).y;
	float dvx = velAt(uv + vec2(0, t.y)).x - velAt(uv - vec2(0, t.y)).x;
	return (dvy - dvx) / (2.0 * busCell());
}

out vec4 fragColor;
void main()
{
	vec2 uv = vUV.st;
	float dt = BUS_DT;
	vec4 obstacle = texture(sTD2DInputs[3], uv);
	if (busSolid(obstacle))
	{
		fragColor = TDOutputSwizzle(vec4(obstacle.gb, 0.0, 0.0));
		return;
	}
	vec2 v = texture(sTD2DInputs[0], uv).xy;
	vec4 s = texture(sTD2DInputs[1], uv);
	v.y += dt * (uForce.x * s.g - uForce.y * s.r);
	v += dt * texture(sTD2DInputs[2], uv).xy;
	v += (uWind.xy - v) * (1.0 - exp(-uWind.z * dt));

	if (uForce.z > 0.0)
	{
		vec2 t = busTexel();
		float h = busCell();
		float c = curlAt(uv);
		vec2 eta = vec2(abs(curlAt(uv + vec2(t.x, 0))) - abs(curlAt(uv - vec2(t.x, 0))),
		                abs(curlAt(uv + vec2(0, t.y))) - abs(curlAt(uv - vec2(0, t.y)))) / (2.0 * h);
		vec2 n = eta / (length(eta) + 1e-5);
		v += dt * uForce.z * h * vec2(n.y * c, -n.x * c);
	}
	fragColor = TDOutputSwizzle(vec4(v, 0.0, 0.0));
}
