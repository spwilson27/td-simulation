// Raymarch the particle source volume (heat in g) and add it over a volume render.
// in0 is a dummy 2D input. The source volume is sTD3DInputs[0].

uniform vec4 uBMin;
uniform vec4 uBMax;
uniform vec4 uCam;
uniform vec4 uTarget;
uniform vec4 uGain;   // x emission scale

vec3 blackbody(float t)
{
	t = clamp(t, 0.0, 1.0);
	vec3 c = vec3(min(t * 1.8, 1.0), pow(t, 1.35) * 0.75, pow(t, 3.2));
	return mix(c, vec3(1.0, 0.92, 0.8), smoothstep(0.65, 1.0, t));
}

out vec4 fragColor;
void main()
{
	vec2 res = uTDOutputInfo.res.zw;
	vec2 ndc = (gl_FragCoord.xy / res) * 2.0 - 1.0;
	float aspect = res.x / max(res.y, 1.0);
	float fov = radians(max(uCam.w, 10.0));
	vec3 target = uTarget.xyz;
	vec3 eye = target + max(uCam.z, 0.2) * vec3(cos(uCam.y) * sin(uCam.x), sin(uCam.y), cos(uCam.y) * cos(uCam.x));
	vec3 forward = normalize(target - eye);
	vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
	vec3 up = cross(right, forward);
	float scale = tan(fov * 0.5);
	vec3 rd = normalize(forward + ndc.x * aspect * scale * right + ndc.y * scale * up);
	vec3 bmin = uBMin.xyz;
	vec3 bmax = uBMax.xyz;
	vec3 inv = 1.0 / max(abs(rd), vec3(1e-4)) * sign(rd + vec3(1e-8));
	vec3 t0 = (bmin - eye) * inv;
	vec3 t1 = (bmax - eye) * inv;
	float tNear = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
	float tFar = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
	if (tFar <= max(tNear, 0.0))
	{
		fragColor = TDOutputSwizzle(vec4(0.0));
		return;
	}
	tNear = max(tNear, 0.0);
	const int kSteps = 24;
	float dt = (tFar - tNear) / float(kSteps);
	vec3 acc = vec3(0.0);
	vec3 size = bmax - bmin;
	for (int i = 0; i < kSteps; i++)
	{
		vec3 p = eye + rd * (tNear + (float(i) + 0.5) * dt);
		vec3 uv = clamp((p - bmin) / size, vec3(0.0), vec3(1.0));
		float heat = texture(sTD3DInputs[0], uv).g;
		acc += blackbody(clamp(heat * 0.35, 0.0, 1.0)) * heat * uGain.x * dt;
	}
	fragColor = TDOutputSwizzle(vec4(acc, 1.0));
}
