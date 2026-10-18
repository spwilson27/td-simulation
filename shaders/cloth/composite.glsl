// in0 cloth render (alpha 1 on the cloth), in1 background (volume or a flat plate).
// No TDOutputSwizzle: numpyArray of a swizzled TOP comes back magenta.
out vec4 fragColor;

void main()
{
	ivec2 p = ivec2(gl_FragCoord.xy);
	vec4 cloth = texelFetch(sTD2DInputs[0], p, 0);
	vec4 bg = texelFetch(sTD2DInputs[1], p, 0);
	float a = clamp(cloth.a, 0.0, 1.0);
	vec3 rgb = mix(bg.rgb, cloth.rgb, a);
	fragColor = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
