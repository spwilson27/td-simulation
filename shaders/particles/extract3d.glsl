// td: output3d
// Read the decoded particle splat volume and write bus forces or sources.
// uKind.x < 0.5 forces (xyz), else sources (smoke, heat). uSrc.x is smoke per unit heat.

uniform vec4 uKind;
uniform vec4 uSrc;

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main()
{
	ivec3 c = ivec3(gl_GlobalInvocationID.xyz);
	ivec3 size = ivec3(uTDOutputInfo.res.zw, uTDOutputInfo.depth.y);
	if (any(greaterThanEqual(c, size)))
		return;
	vec4 s = texelFetch(sTD3DInputs[0], c, 0);
	vec4 outv = uKind.x < 0.5 ? vec4(s.rgb, 0.0) : vec4(s.a * uSrc.x, s.a, 0.0, 0.0);
	TDImageStoreOutput(0, c, outv);
}
