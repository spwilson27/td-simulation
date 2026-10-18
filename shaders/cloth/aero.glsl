#include <cloth_common_glsl>
// Aerodynamic force on each vertex, from the same model the integrator uses.
// 2D in0 pos, in1 vel, 3D in0 wind.
uniform vec4 uBMin;
uniform vec4 uBMax;
uniform vec4 uGrav;
uniform vec4 uAero;
uniform vec4 uLen;
layout(local_size_x = 8, local_size_y = 8) in;

void main()
{
	ivec2 size = ivec2(uTDOutputInfo.res.zw);
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	if (coord.x >= size.x || coord.y >= size.y)
		return;
	vec4 pos = texelFetch(sTD2DInputs[0], coord, 0);
	vec3 v = texelFetch(sTD2DInputs[1], coord, 0).xyz;
	vec3 force = vec3(0.0);
	if (pos.w > 0.0 && uGrav.w > 0.0)
	{
		vec3 span = max(uBMax.xyz - uBMin.xyz, vec3(1e-6));
		vec3 uv = clamp((pos.xyz - uBMin.xyz) / span, vec3(0.0), vec3(1.0));
		vec3 n = clothNormal(sTD2DInputs[0], coord, size);
		force = clothAero(n, v, texture(sTD3DInputs[0], uv).xyz, uLen.x * uLen.y, uGrav.w, uAero.x, uAero.y);
	}
	TDImageStoreOutput(0, ivec3(coord, 0), vec4(force, pos.w));
}
