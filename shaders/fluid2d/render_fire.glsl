#include <bus_glsl>
// Display: smoke absorbs, hot gas emits along an approximate blackbody ramp. Obstacles are drawn as a flat grey.
// in0: scalars, in1: obstacles
uniform vec4 uLook;       // x smoke density, y fire intensity, z temperature at full white-hot, w obstacle opacity
uniform vec4 uSmokeColor; // rgb smoke albedo, a background brightness

vec3 blackbody(float t)
{
	t = clamp(t, 0.0, 1.0);
	return clamp(vec3(t * 3.0, t * 3.0 - 1.0, t * 3.0 - 2.0), 0.0, 1.0) * vec3(1.0, 0.85, 0.7);
}

out vec4 fragColor;
void main()
{
	vec2 uv = vUV.st;
	vec4 s = texture(sTD2DInputs[0], uv);
	float smoke = 1.0 - exp(-max(s.r, 0.0) * uLook.x);
	float t = max(s.g, 0.0) / max(uLook.z, 1e-4);
	vec3 col = mix(vec3(uSmokeColor.a), uSmokeColor.rgb, smoke);
	col += blackbody(t) * uLook.y * smoothstep(0.05, 0.3, t);
	if (busSolid(texture(sTD2DInputs[1], uv)))
		col = mix(col, vec3(0.35), uLook.w);
	fragColor = TDOutputSwizzle(vec4(col, 1.0));
}
