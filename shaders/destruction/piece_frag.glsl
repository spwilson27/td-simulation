// Exterior is a lit concrete, interior (the cut) is a warmer face so the fracture reads.
in vec3 vNormal;
in vec3 vWorld;
in float vInterior;

uniform vec4 uLight;

out vec4 fragColor;

void main()
{
	vec3 n = normalize(vNormal);
	vec3 l = normalize(uLight.xyz);
	float ndl = clamp(dot(n, l), 0.0, 1.0);
	float wrap = clamp(dot(n, l) * 0.5 + 0.5, 0.0, 1.0);
	vec3 exterior = vec3(0.42, 0.44, 0.46);
	vec3 interior = vec3(0.55, 0.32, 0.18);
	vec3 base = mix(exterior, interior, step(0.5, vInterior));
	vec3 col = base * (0.18 + 0.82 * mix(ndl, wrap, 0.35));
	float rim = pow(1.0 - abs(dot(n, normalize(uLight.xyz))), 2.0);
	col += interior * rim * 0.08 * step(0.5, vInterior);
	fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
