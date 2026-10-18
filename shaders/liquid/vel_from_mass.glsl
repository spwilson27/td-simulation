// Velocity from gathered momentum. w keeps the mass so later passes still know it.
// td: output3d
uniform vec4 uUnused;
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main()
{
	ivec3 c = ivec3(gl_GlobalInvocationID.xyz);
	ivec3 res = ivec3(uTDOutputInfo.res.zw, uTDOutputInfo.depth.y);
	if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, res)))
		return;
	vec4 m = texelFetch(sTD3DInputs[0], c, 0);
	vec3 v = m.a > 1e-12 ? m.xyz / m.a : vec3(0.0);
	TDImageStoreOutput(0, c, vec4(v, m.a));
}
