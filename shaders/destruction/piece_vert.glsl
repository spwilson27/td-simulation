// td: stage vertex
// One merged mesh. Cd.r is the piece index, Cd.g > 0.5 is an interior (cut) face.
// sState / sExtents are the rigid textures. The orbit matches shaders/fluid3d/render_volume.glsl.
uniform sampler2D sState;
uniform vec4 uCount;
uniform vec4 uCam;
uniform vec4 uTarget;
uniform vec4 uRes;

out vec3 vNormal;
out vec3 vWorld;
out float vInterior;

int flipY(int y, int height)
{
	return uCount.w > 0.5 ? (height - 1 - y) : y;
}

ivec2 statePix(int id, int quad)
{
	ivec2 size = textureSize(sState, 0);
	int x = id % 128;
	int y = id / 128;
	if (quad == 1 || quad == 3) x += 128;
	if (quad == 2 || quad == 3) y += 128;
	return ivec2(x, flipY(y, size.y));
}

vec3 qrot(vec4 q, vec3 v)
{
	return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

void main()
{
	int id = int(Cd.r + 0.5);
	vec3 center = texelFetch(sState, statePix(id, 0), 0).xyz;
	vec4 quat = texelFetch(sState, statePix(id, 1), 0);
	vec3 world = qrot(quat, TDPos()) + center;
	vWorld = world;
	vNormal = normalize(qrot(quat, N));
	vInterior = Cd.g;
	vec3 target = uTarget.xyz;
	float az = uCam.x;
	float el = uCam.y;
	vec3 eye = target + max(uCam.z, 0.2) * vec3(cos(el) * sin(az), sin(el), cos(el) * cos(az));
	vec3 forward = normalize(target - eye);
	vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
	vec3 up = cross(right, forward);
	vec3 rel = world - eye;
	float z = max(dot(rel, forward), 1e-3);
	float aspect = uRes.x / max(uRes.y, 1.0);
	float scale = tan(radians(max(uCam.w, 10.0)) * 0.5);
	vec2 ndc = vec2(dot(rel, right) / (z * aspect * scale), dot(rel, up) / (z * scale));
	gl_Position = vec4(ndc * z, z * 0.5, z);
}
