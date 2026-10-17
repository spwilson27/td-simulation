#include <bus3d_glsl>
// Buoyancy, bus forces, wind drag, vorticity confinement.
// in0 velocity, in1 scalars, in2 forces, in3 obstacles.
uniform vec4 uForce;  // x buoyancy, y weight, z vorticity
uniform vec4 uWind;   // xyz wind velocity, w drag (1/s)
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

vec3 velAt(vec3 uv)
{
	return texture(sTD3DInputs[0], busClampUV(uv)).xyz;
}

vec3 curlAt(vec3 uv)
{
	vec3 t = busTexel();
	float h2 = 2.0 * busCell();
	vec3 px = velAt(uv + vec3(t.x, 0, 0)) - velAt(uv - vec3(t.x, 0, 0));
	vec3 py = velAt(uv + vec3(0, t.y, 0)) - velAt(uv - vec3(0, t.y, 0));
	vec3 pz = velAt(uv + vec3(0, 0, t.z)) - velAt(uv - vec3(0, 0, t.z));
	return vec3(py.z - pz.y, pz.x - px.z, px.y - py.x) / h2;
}

void main()
{
	ivec3 c = busCoord();
	if (!busLive(c))
		return;
	vec3 uv = busUV(c);
	vec4 obstacle = texture(sTD3DInputs[3], uv);
	if (busSolid(obstacle))
	{
		TDImageStoreOutput(0, c, vec4(busObstacleVel(obstacle), 0.0));
		return;
	}
	float dt = BUS_DT;
	vec3 v = texelFetch(sTD3DInputs[0], c, 0).xyz;
	vec4 s = texelFetch(sTD3DInputs[1], c, 0);
	v.y += dt * (uForce.x * s.g - uForce.y * s.r);
	v += dt * texture(sTD3DInputs[2], uv).xyz;
	v += (uWind.xyz - v) * (1.0 - exp(-uWind.w * dt));
	// The obstacle surface is a one-cell vortex sheet. Confinement there explodes, so fade it out
	// inside a couple of cells of a solid.
	float gate = smoothstep(0.0, 2.5 * busCell(), obstacle.r);
	if (uForce.z > 0.0 && gate > 0.0)
	{
		vec3 t = busTexel();
		float h = busCell();
		vec3 w = curlAt(uv);
		float wpx = length(curlAt(uv + vec3(t.x, 0, 0)));
		float wmx = length(curlAt(uv - vec3(t.x, 0, 0)));
		float wpy = length(curlAt(uv + vec3(0, t.y, 0)));
		float wmy = length(curlAt(uv - vec3(0, t.y, 0)));
		float wpz = length(curlAt(uv + vec3(0, 0, t.z)));
		float wmz = length(curlAt(uv - vec3(0, 0, t.z)));
		vec3 eta = vec3(wpx - wmx, wpy - wmy, wpz - wmz) / (2.0 * h);
		vec3 n = eta / (length(eta) + 1e-5);
		v += dt * uForce.z * h * cross(n, w) * gate;
	}
	TDImageStoreOutput(0, c, vec4(v, 0.0));
}
