// A fan: constant acceleration inside a world-space box, zero outside.
// The fluid carries this as a jet. The cloth samples the resulting bus velocity.
// td: output3d
uniform vec4 uBMin;
uniform vec4 uBMax;
uniform vec4 uBox0; // xyz min, w acceleration along +Z
uniform vec4 uBox1; // xyz max
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

void main()
{
	ivec3 c = ivec3(gl_GlobalInvocationID.xyz);
	ivec3 res = ivec3(int(uTDOutputInfo.res.z), int(uTDOutputInfo.res.w), int(uTDOutputInfo.depth.y + 0.5));
	if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, res)))
		return;
	vec3 span = max(uBMax.xyz - uBMin.xyz, vec3(1e-6));
	vec3 p = uBMin.xyz + (vec3(c) + 0.5) * span / vec3(res);
	float acc = 0.0;
	if (all(greaterThanEqual(p, uBox0.xyz)) && all(lessThanEqual(p, uBox1.xyz)))
		acc = uBox0.w;
	TDImageStoreOutput(0, c, vec4(0.0, 0.0, acc, 0.0));
}
