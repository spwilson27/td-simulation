// Debris over the combined composite. in0 is the opaque scene, in1 is the particle beauty
// (additive colour on black). A fifth input on the scene composite does not compile here.

out vec4 fragColor;
void main()
{
	ivec2 p = ivec2(gl_FragCoord.xy);
	vec3 base = texelFetch(sTD2DInputs[0], p, 0).rgb;
	vec3 dust = texelFetch(sTD2DInputs[1], p, 0).rgb;
	fragColor = vec4(clamp(base + dust, 0.0, 1.5), 1.0);
}
