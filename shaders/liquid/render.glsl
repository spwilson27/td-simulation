// Raymarched liquid surface. in0 phi (negative inside), in1 solids, in2 background plate (2D).
// The plate is a flat colour unless a scene composite is switched in. Output alpha is 1.
uniform vec4 uBMin;
uniform vec4 uBMax;
uniform vec4 uCam;
uniform vec4 uTarget;
uniform vec4 uLight;
uniform vec4 uRay;
uniform vec4 uLook; // x absorption, y specular, z fresnel, w unused

vec3 worldToUV(vec3 p)
{
	return (p - uBMin.xyz) / max(uBMax.xyz - uBMin.xyz, vec3(1e-6));
}

float scenePhi(vec3 uv)
{
	float liquid = texture(sTD3DInputs[0], clamp(uv, vec3(0.0), vec3(1.0))).r;
	float solid = texture(sTD3DInputs[1], clamp(uv, vec3(0.0), vec3(1.0))).r;
	// Draw the solid when it is closer than the liquid surface (more negative, or liquid is outside).
	if (solid < liquid)
		return solid;
	return liquid;
}

vec3 sceneGrad(vec3 uv)
{
	vec3 t = 1.0 / vec3(textureSize(sTD3DInputs[0], 0));
	float dx = scenePhi(clamp(uv + vec3(t.x, 0, 0), 0.0, 1.0)) - scenePhi(clamp(uv - vec3(t.x, 0, 0), 0.0, 1.0));
	float dy = scenePhi(clamp(uv + vec3(0, t.y, 0), 0.0, 1.0)) - scenePhi(clamp(uv - vec3(0, t.y, 0), 0.0, 1.0));
	float dz = scenePhi(clamp(uv + vec3(0, 0, t.z), 0.0, 1.0)) - scenePhi(clamp(uv - vec3(0, 0, t.z), 0.0, 1.0));
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
	float tNear = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
	float tFar = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
	vec3 bg = texture(sTD2DInputs[0], clamp(gl_FragCoord.xy / res, vec2(0.0), vec2(1.0))).rgb;
	if (tFar <= max(tNear, 0.0))
	{
		fragColor = TDOutputSwizzle(vec4(bg, 1.0));
		return;
	}
	tNear = max(tNear, 0.0);
	const int kMaxSteps = 48;
	int steps = clamp(int(uRay.x + 0.5), 12, kMaxSteps);
	float dt = (tFar - tNear) / float(steps);
	float t = tNear;
	vec3 light = normalize(uLight.xyz);
	vec3 col = bg;
	bool hit = false;
	for (int i = 0; i < kMaxSteps; i++)
	{
		if (i >= steps || t > tFar)
			break;
		vec3 p = eye + rd * t;
		vec3 uv = worldToUV(p);
		if (all(greaterThanEqual(uv, vec3(0.0))) && all(lessThanEqual(uv, vec3(1.0))))
		{
			float sd = scenePhi(uv);
			if (sd < 0.0)
			{
				// Step back a little so the normal sits on the surface rather than deep inside.
				p -= rd * min(dt, max(-sd, 0.0));
				uv = clamp(worldToUV(p), vec3(0.0), vec3(1.0));
				vec3 n = normalize(sceneGrad(uv) + vec3(0.0, 1e-4, 0.0));
				if (dot(n, rd) > 0.0)
					n = -n;
				float solid = texture(sTD3DInputs[1], uv).r;
				float liquid = texture(sTD3DInputs[0], uv).r;
				bool isSolid = solid < liquid;
				float ndl = clamp(dot(n, light), 0.0, 1.0);
				float fres = pow(1.0 - clamp(dot(n, -rd), 0.0, 1.0), 3.0);
				vec3 ref = reflect(rd, n);
				vec2 plateUV = clamp(gl_FragCoord.xy / res + ref.xy * 0.04, vec2(0.0), vec2(1.0));
				vec3 through = texture(sTD2DInputs[0], plateUV).rgb;
				vec3 water = vec3(0.03, 0.18, 0.28);
				float depth = clamp(-liquid * 4.0, 0.0, 1.0);
				water = mix(water, vec3(0.02, 0.08, 0.16), depth);
				vec3 lit = water * (0.22 + 0.78 * ndl);
				float spec = pow(clamp(dot(ref, light), 0.0, 1.0), 40.0) * uLook.y;
				vec3 surf = isSolid ? vec3(0.45, 0.46, 0.5) * (0.25 + 0.75 * ndl) : mix(lit, through, 0.25);
				surf += spec * vec3(0.9, 0.95, 1.0);
				surf = mix(surf, vec3(0.75, 0.85, 0.9), isSolid ? 0.0 : fres * uLook.z);
				col = surf;
				hit = true;
				break;
			}
		}
		t += dt;
	}
	if (!hit)
		col = bg;
	fragColor = TDOutputSwizzle(vec4(clamp(col, 0.0, 1.0), 1.0));
}
