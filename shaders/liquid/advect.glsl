#include <liquid_common_glsl>
// Advance positions and keep them out of the closed box and out of real solids (alpha < 0.5).
// in0 pos, in1 new velocity, 3D in0 obstacles.
uniform vec4 uBMin;
uniform vec4 uBMax;
uniform vec4 uClock;
uniform vec4 uCollide; // x radius
layout(local_size_x = 8, local_size_y = 8) in;

vec3 obsGrad(vec3 uv)
{
	vec3 t = 1.0 / vec3(textureSize(sTD3DInputs[0], 0));
	float dx = texture(sTD3DInputs[0], clamp(uv + vec3(t.x, 0, 0), 0.0, 1.0)).r
	         - texture(sTD3DInputs[0], clamp(uv - vec3(t.x, 0, 0), 0.0, 1.0)).r;
	float dy = texture(sTD3DInputs[0], clamp(uv + vec3(0, t.y, 0), 0.0, 1.0)).r
	         - texture(sTD3DInputs[0], clamp(uv - vec3(0, t.y, 0), 0.0, 1.0)).r;
	float dz = texture(sTD3DInputs[0], clamp(uv + vec3(0, 0, t.z), 0.0, 1.0)).r
	         - texture(sTD3DInputs[0], clamp(uv - vec3(0, 0, t.z), 0.0, 1.0)).r;
	return vec3(dx, dy, dz);
}

void main()
{
	ivec2 size = ivec2(uTDOutputInfo.res.zw);
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	if (coord.x >= size.x || coord.y >= size.y)
		return;
	vec4 pos = texelFetch(sTD2DInputs[0], coord, 0);
	vec3 vel = texelFetch(sTD2DInputs[1], coord, 0).xyz;
	if (pos.a <= 0.0)
	{
		TDImageStoreOutput(0, ivec3(coord, 0), pos);
		return;
	}
	vec3 p = pos.xyz + vel * uClock.x;
	float radius = max(uCollide.x, 0.0);
	vec3 lo = uBMin.xyz + radius;
	vec3 hi = uBMax.xyz - radius;
	vec3 before = p;
	p = clamp(p, lo, hi);
	// Kill the component that was trying to leave.
	vec3 pushed = p - before;
	vel += pushed / max(uClock.x, 1e-5);
	vec3 sizew = max(uBMax.xyz - uBMin.xyz, vec3(1e-6));
	vec3 uv = clamp((p - uBMin.xyz) / sizew, vec3(0.0), vec3(1.0));
	vec4 obstacle = texture(sTD3DInputs[0], uv);
	if (obstacle.r < radius)
	{
		vec3 g = obsGrad(uv);
		float glen = length(g);
		vec3 n = glen > 1e-6 ? g / glen : vec3(0.0, 1.0, 0.0);
		p += n * (radius - obstacle.r);
		float radial = dot(vel, n);
		if (radial < 0.0)
			vel -= radial * n;
		vel = 0.5 * vel + 0.5 * obstacle.gba;
		p = clamp(p, lo, hi);
	}
	TDImageStoreOutput(0, ivec3(coord, 0), vec4(p, pos.a));
}
