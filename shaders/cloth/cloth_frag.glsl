// Two-sided cloth. Front and back keep their own colour; the light wraps so neither face is black.
in vec3 vNormal;
in vec3 vWorld;

uniform vec4 uFront;
uniform vec4 uBack;
uniform vec4 uLight;

out vec4 fragColor;

void main()
{
	vec3 n = normalize(vNormal);
	if (!gl_FrontFacing)
		n = -n;
	vec3 base = gl_FrontFacing ? uFront.rgb : uBack.rgb;
	vec3 l = normalize(uLight.xyz);
	float ndl = clamp(dot(n, l) * 0.65 + 0.35, 0.0, 1.0);
	float lift = clamp(vWorld.y * 0.04, -0.04, 0.06);
	vec3 col = base * (0.18 + 0.82 * ndl) + vec3(lift);
	fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
