// Raymarched smoke and fire. in0 scalars, in1 obstacles, in2 shadow transmittance.
// Not including bus3d.glsl: busRes/busTexel read uTDOutputInfo.depth, which a 2D TOP does not have.
uniform vec4 uBMin;
uniform vec4 uBMax;
uniform vec4 uClock;
uniform vec4 uClock2;
uniform vec4 uCam;      // azimuth, elevation, distance, fov degrees
uniform vec4 uTarget;
uniform vec4 uLight;
uniform vec4 uLook;     // smoke density, fire intensity, white-hot temperature, obstacle shade
uniform vec4 uSmoke;    // rgb albedo, a background
uniform vec4 uRay;      // x step count

vec3 busSize() { return uBMax.xyz - uBMin.xyz; }
vec3 busWorldToUV(vec3 p) { return (p - uBMin.xyz) / busSize(); }
bool busOutside(vec3 uv) { return any(lessThan(uv, vec3(0.0))) || any(greaterThan(uv, vec3(1.0))); }

vec3 blackbody(float t)
{
	t = clamp(t, 0.0, 1.0);
	vec3 c = vec3(min(t * 1.8, 1.0), pow(t, 1.35) * 0.75, pow(t, 3.2));
	return mix(c, vec3(1.0, 0.92, 0.8), smoothstep(0.65, 1.0, t));
}

float hash12(vec2 p)
{
	vec3 p3 = fract(vec3(p.xyx) * 0.1031);
	p3 += dot(p3, p3.yzx + 33.33);
	return fract((p3.x + p3.y) * p3.z);
}

vec3 sdfGrad(vec3 uv)
{
	vec3 t = 1.0 / vec3(textureSize(sTD3DInputs[1], 0));
	float dx = texture(sTD3DInputs[1], clamp(uv + vec3(t.x, 0, 0), 0.0, 1.0)).r
	         - texture(sTD3DInputs[1], clamp(uv - vec3(t.x, 0, 0), 0.0, 1.0)).r;
	float dy = texture(sTD3DInputs[1], clamp(uv + vec3(0, t.y, 0), 0.0, 1.0)).r
	         - texture(sTD3DInputs[1], clamp(uv - vec3(0, t.y, 0), 0.0, 1.0)).r;
	float dz = texture(sTD3DInputs[1], clamp(uv + vec3(0, 0, t.z), 0.0, 1.0)).r
	         - texture(sTD3DInputs[1], clamp(uv - vec3(0, 0, t.z), 0.0, 1.0)).r;
	return vec3(dx, dy, dz);
}

out vec4 fragColor;
void main()
{
	vec2 res = uTDOutputInfo.res.zw;
	vec2 ndc = (gl_FragCoord.xy / res) * 2.0 - 1.0;
	float aspect = res.x / max(res.y, 1.0);
	float fov = radians(max(uCam.w, 10.0));
	vec3 target = uTarget.xyz;
	float az = uCam.x;
	float el = uCam.y;
	vec3 eye = target + max(uCam.z, 0.2) * vec3(cos(el) * sin(az), sin(el), cos(el) * cos(az));
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
	vec3 tsm = min(t0, t1);
	vec3 tbg = max(t0, t1);
	float tNear = max(max(tsm.x, tsm.y), tsm.z);
	float tFar = min(min(tbg.x, tbg.y), tbg.z);
	vec3 bg = vec3(uSmoke.a);
	if (tFar <= max(tNear, 0.0))
	{
		fragColor = TDOutputSwizzle(vec4(bg, 1.0));
		return;
	}
	tNear = max(tNear, 0.0);
	// Fixed bound so the Metal compiler can unroll; uRay.x selects how many of those steps run.
	const int kMaxSteps = 56;
	int steps = clamp(int(uRay.x + 0.5), 16, kMaxSteps);
	float dt = (tFar - tNear) / float(steps);
	float t = tNear + dt * (0.15 + 0.7 * hash12(gl_FragCoord.xy));
	vec3 light = normalize(uLight.xyz + vec3(1e-4, 1.0, 1e-4));
	vec3 col = vec3(0.0);
	float trans = 1.0;
	for (int i = 0; i < kMaxSteps; i++)
	{
		if (i >= steps || trans < 0.015 || t > tFar)
			break;
		vec3 p = eye + rd * t;
		vec3 uv = busWorldToUV(p);
		if (!busOutside(uv))
		{
			float sd = texture(sTD3DInputs[1], uv).r;
			if (sd < 0.0)
			{
				vec3 n = normalize(sdfGrad(uv) + vec3(0.0, 1e-4, 0.0));
				float ndl = clamp(dot(n, light), 0.0, 1.0);
				float sh = clamp(texture(sTD3DInputs[2], uv).r, 0.0, 1.0);
				vec3 surf = vec3(0.42, 0.44, 0.48) * (0.22 + 0.78 * ndl) * mix(0.4, 1.0, sh);
				col += trans * surf * uLook.w;
				break;
			}
			vec4 s = texture(sTD3DInputs[0], uv);
			float dens = max(s.r, 0.0);
			float temp = max(s.g, 0.0);
			float sh = clamp(texture(sTD3DInputs[2], uv).r, 0.0, 1.0);
			float absorb = dens * uLook.x * dt;
			float stepT = exp(-absorb);
			vec3 emit = (blackbody(temp / max(uLook.z, 0.2)) * uLook.y * temp + uSmoke.rgb * dens * 0.9) * sh;
			col += trans * emit * (1.0 - stepT);
			trans *= stepT;
		}
		t += dt;
	}
	col = col + bg * trans;
	fragColor = TDOutputSwizzle(vec4(clamp(col, 0.0, 4.0), 1.0));
}
