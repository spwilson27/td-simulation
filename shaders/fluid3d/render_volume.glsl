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
uniform vec4 uDetail;   // strength, frequency, advect, unused. Strength 0 leaves the cached field unchanged.

// texture() on these 3D inputs in a pixel shader follows the mip chain. The volumes have no
// mips, and the missing level comes back as the checker. texelFetch stays on level 0.
vec4 volTexel(sampler3D tex, vec3 uv)
{
	ivec3 size = textureSize(tex, 0);
	ivec3 ijk = clamp(ivec3(clamp(uv, vec3(0.0), vec3(1.0)) * vec3(size)), ivec3(0), size - ivec3(1));
	return texelFetch(tex, ijk, 0);
}

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

float hash13(vec3 p)
{
	p = fract(p * 0.1031);
	p += dot(p, p.yzx + 33.33);
	return fract((p.x + p.y) * p.z);
}

float valueNoise(vec3 p)
{
	vec3 i = floor(p);
	vec3 f = fract(p);
	vec3 u = f * f * (3.0 - 2.0 * f);
	return mix(mix(mix(hash13(i), hash13(i + vec3(1, 0, 0)), u.x),
	               mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), u.x), u.y),
	           mix(mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), u.x),
	               mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), u.x), u.y), u.z);
}

// Zero-mean fbm. Energy (smoke times temperature) keeps empty cells empty, so the plume's
// large-scale shape stays while the octaves add the high frequency a coarser cache does not store.
float upresDensity(vec3 uv, float dens, float temp)
{
	if (uDetail.x <= 0.0)
		return dens;
	vec3 flow = vec3(0.0);
	if (TD_NUM_3D_INPUTS > 3)
		flow = volTexel(sTD3DInputs[3], uv).xyz * uDetail.z;
	vec3 q = uv * max(uDetail.y, 1.0) + flow * uClock.y;
	// Warp the sample across the plume. A zero-mean displacement moves edges without shifting
	// the low-passed shape, and it is not averaged away by the ray integral the way additive noise is.
	vec3 warp = vec3(
		valueNoise(q),
		valueNoise(q + vec3(4.2, 1.7, 0.3)),
		valueNoise(q + vec3(1.1, 8.4, 2.9))) - 0.5;
	float amp = uDetail.x * 0.035 * (0.55 + 0.2 * temp);
	vec3 uv2 = clamp(uv + warp * amp, vec3(0.0), vec3(1.0));
	float warped = max(volTexel(sTD3DInputs[0], uv2).r, 0.0);
	// Second octave, multiplied into the smoke that was actually sampled. Empty cells stay empty,
	// and this never touches the obstacle branch.
	float hi = valueNoise(q * 2.7 + vec3(3.1, 0.4, 1.7)) - 0.5;
	float hi2 = valueNoise(q * 5.3 + vec3(0.2, 4.8, 2.2)) - 0.5;
	float modulate = 1.0 + uDetail.x * (0.55 * hi + 0.28 * hi2);
	return warped * max(modulate, 0.0);
}

vec3 sdfGrad(vec3 uv)
{
	vec3 t = 1.0 / vec3(textureSize(sTD3DInputs[1], 0));
	float dx = volTexel(sTD3DInputs[1], clamp(uv + vec3(t.x, 0, 0), 0.0, 1.0)).r
	         - volTexel(sTD3DInputs[1], clamp(uv - vec3(t.x, 0, 0), 0.0, 1.0)).r;
	float dy = volTexel(sTD3DInputs[1], clamp(uv + vec3(0, t.y, 0), 0.0, 1.0)).r
	         - volTexel(sTD3DInputs[1], clamp(uv - vec3(0, t.y, 0), 0.0, 1.0)).r;
	float dz = volTexel(sTD3DInputs[1], clamp(uv + vec3(0, 0, t.z), 0.0, 1.0)).r
	         - volTexel(sTD3DInputs[1], clamp(uv - vec3(0, 0, t.z), 0.0, 1.0)).r;
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
			float sd = volTexel(sTD3DInputs[1], uv).r;
			if (sd < 0.0)
			{
				vec3 n = normalize(sdfGrad(uv) + vec3(0.0, 1e-4, 0.0));
				float ndl = clamp(dot(n, light), 0.0, 1.0);
				float sh = clamp(volTexel(sTD3DInputs[2], uv).r, 0.0, 1.0);
				vec3 surf = vec3(0.42, 0.44, 0.48) * (0.22 + 0.78 * ndl) * mix(0.4, 1.0, sh);
				col += trans * surf * uLook.w;
				break;
			}
			vec4 s = volTexel(sTD3DInputs[0], uv);
			float temp = max(s.g, 0.0);
			float dens = upresDensity(uv, max(s.r, 0.0), temp);
			float sh = clamp(volTexel(sTD3DInputs[2], uv).r, 0.0, 1.0);
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
