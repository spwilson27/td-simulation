// out = -in0. Used to feed the V-cycle the preconditioner right-hand side.
// td: output3d
uniform vec4 uUnused;
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main()
{
	ivec3 c = ivec3(gl_GlobalInvocationID.xyz);
	ivec3 res = ivec3(uTDOutputInfo.res.zw, uTDOutputInfo.depth.y);
	if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, res)))
		return;
	TDImageStoreOutput(0, c, vec4(-texelFetch(sTD3DInputs[0], c, 0).r, 0.0, 0.0, 1.0));
}
