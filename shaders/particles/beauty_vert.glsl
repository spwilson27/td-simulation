// td: stage vertex
// Point sprites. 2D maps the bus rectangle onto the frame. 3D uses the same orbit camera as the volume renderer.
// sState is position.xyz + age, sVel is velocity.xyz + life remaining.

uniform sampler2D sState;
uniform sampler2D sVel;
uniform vec4 uMode;     // x > 0.5 for the 3D orbit camera
uniform vec4 uPoint;    // x pixel size
uniform vec4 uLook;     // x colour scale
uniform vec4 uBounds;   // 2D xmin ymin xmax ymax
uniform vec4 uBMin;
uniform vec4 uBMax;
uniform vec4 uCam;      // azimuth, elevation, distance, fov degrees
uniform vec4 uTarget;
uniform vec4 uRes;      // render width, height

out vec4 vColor;

void main()
{
	ivec2 size = textureSize(sState, 0);
	int id = TDInstanceID();
	ivec2 coord = ivec2(id % max(size.x, 1), id / max(size.x, 1));
	vec4 state = texelFetch(sState, coord, 0);
	vec3 pos = state.xyz;
	float age = state.w;
	float life = texelFetch(sVel, coord, 0).w;
	float span = max(age + life, 1e-3);
	float u = clamp(age / span, 0.0, 1.0);
	vec3 hot = mix(vec3(1.0, 0.96, 0.8), vec3(1.0, 0.38, 0.04), smoothstep(0.05, 0.55, u));
	hot = mix(hot, vec3(0.35, 0.03, 0.0), smoothstep(0.5, 1.0, u));
	float alpha = (1.0 - u) * (1.0 - u);
	vColor = vec4(hot * uLook.x, alpha);
	if (life <= 0.0)
	{
		gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
		gl_PointSize = 1.0;
		return;
	}
	vec2 ndc;
	if (uMode.x > 0.5)
	{
		vec3 target = uTarget.xyz;
		float az = uCam.x;
		float el = uCam.y;
		vec3 eye = target + max(uCam.z, 0.2) * vec3(cos(el) * sin(az), sin(el), cos(el) * cos(az));
		vec3 forward = normalize(target - eye);
		vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
		vec3 up = cross(right, forward);
		vec3 rel = pos - eye;
		float z = dot(rel, forward);
		if (z < 0.05)
		{
			gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
			gl_PointSize = 1.0;
			return;
		}
		float aspect = uRes.x / max(uRes.y, 1.0);
		float f = 1.0 / tan(radians(max(uCam.w, 10.0)) * 0.5);
		ndc = vec2(dot(rel, right) / z * f / aspect, dot(rel, up) / z * f);
	}
	else
	{
		vec2 uv = (pos.xy - uBounds.xy) / max(uBounds.zw - uBounds.xy, vec2(1e-6));
		ndc = uv * 2.0 - 1.0;
	}
	gl_Position = vec4(ndc, 0.5, 1.0);
	gl_PointSize = max(uPoint.x, 1.0);
}
