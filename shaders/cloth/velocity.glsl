#include <cloth_common_glsl>
// v = (new pos - previous pos) / dt, or that displacement added to the incoming velocity
// when uMode.x > 0 (the self-collision correction on top of the last substep).
// w accumulates torn structural edges: bit 0 horizontal, bit 1 vertical.
uniform vec4 uDyn;
uniform vec4 uLen;
uniform vec4 uTear;
uniform vec4 uMode;
layout(local_size_x = 8, local_size_y = 8) in;

void main()
{
	ivec2 size = ivec2(uTDOutputInfo.res.zw);
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	if (coord.x >= size.x || coord.y >= size.y)
		return;
	vec3 p = texelFetch(sTD2DInputs[0], coord, 0).xyz;
	vec3 prev = texelFetch(sTD2DInputs[1], coord, 0).xyz;
	vec4 old = texelFetch(sTD2DInputs[2], coord, 0);
	float dt = max(uDyn.x, 1e-6);
	vec3 v = (p - prev) / dt;
	if (uMode.x > 0.5)
		v += old.xyz;
	float speed = length(v);
	if (uDyn.z > 0.0 && speed > uDyn.z)
		v *= uDyn.z / speed;
	int bits = int(old.w + 0.5);
	if (uTear.x > 0.0)
	{
		if (coord.x + 1 < size.x)
		{
			float len = distance(p, texelFetch(sTD2DInputs[0], coord + ivec2(1, 0), 0).xyz);
			if (len > uLen.x * (1.0 + uTear.x))
				bits |= 1;
		}
		if (coord.y + 1 < size.y)
		{
			float len = distance(p, texelFetch(sTD2DInputs[0], coord + ivec2(0, 1), 0).xyz);
			if (len > uLen.y * (1.0 + uTear.x))
				bits |= 2;
		}
	}
	TDImageStoreOutput(0, ivec3(coord, 0), vec4(v, float(bits)));
}
