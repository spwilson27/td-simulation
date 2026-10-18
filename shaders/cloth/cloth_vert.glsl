// td: stage vertex
// Grid vertices carry their indices in Cd. The position texture supplies the world position.
uniform sampler2D sPos;
uniform vec4 uCam;
uniform vec4 uTarget;
uniform vec4 uRes;

out vec3 vNormal;
out vec3 vWorld;

void main()
{
	ivec2 size = textureSize(sPos, 0);
	ivec2 coord = clamp(ivec2(int(Cd.r + 0.5), int(Cd.g + 0.5)), ivec2(0), size - ivec2(1));
	vec3 world = texelFetch(sPos, coord, 0).xyz;
	ivec2 px = clamp(coord + ivec2(1, 0), ivec2(0), size - ivec2(1));
	ivec2 mx = clamp(coord - ivec2(1, 0), ivec2(0), size - ivec2(1));
	ivec2 py = clamp(coord + ivec2(0, 1), ivec2(0), size - ivec2(1));
	ivec2 my = clamp(coord - ivec2(0, 1), ivec2(0), size - ivec2(1));
	vec3 n = cross(texelFetch(sPos, px, 0).xyz - texelFetch(sPos, mx, 0).xyz,
		texelFetch(sPos, py, 0).xyz - texelFetch(sPos, my, 0).xyz);
	vNormal = length(n) > 1e-8 ? normalize(n) : vec3(0.0, 0.0, 1.0);
	vWorld = world;
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
	// A constant window z makes overlapping folds fight. Nearer cloth has to win.
	float ndcZ = clamp((z - 0.05) / 8.0, 0.0, 1.0);
	gl_Position = vec4(ndc * z, ndcZ * z, z);
}
