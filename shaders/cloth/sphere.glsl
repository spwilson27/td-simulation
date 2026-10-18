// Analytic sphere obstacle: signed distance and rigid velocity. Radius 0 is empty space.
uniform vec4 uBMin;
uniform vec4 uBMax;
uniform vec4 uCenter; // xyz centre, w radius
uniform vec4 uMotion; // xyz velocity
uniform vec4 uTime;   // x seconds
// td: output3d
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

void main()
{
	ivec3 c = ivec3(gl_GlobalInvocationID.xyz);
	ivec3 res = ivec3(int(uTDOutputInfo.res.z), int(uTDOutputInfo.res.w), int(uTDOutputInfo.depth.y + 0.5));
	if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, res)))
		return;
	vec3 span = max(uBMax.xyz - uBMin.xyz, vec3(1e-6));
	vec3 p = uBMin.xyz + (vec3(c) + 0.5) * span / vec3(res);
	vec3 centre = uCenter.xyz + uMotion.xyz * uTime.x;
	float phi = uCenter.w > 0.0 ? length(p - centre) - uCenter.w : 1.0e4;
	TDImageStoreOutput(0, c, vec4(phi, uMotion.xyz));
}
