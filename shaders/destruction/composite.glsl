// in0 pieces, in1 volume picture, in2 optional particle beauty.
// The piece render clears to opaque black, so alpha cannot mask it. A near-black
// piece pixel is the background and shows the volume. The result is opaque.
// No TDOutputSwizzle: with it, numpyArray of this TOP comes back magenta.
uniform vec4 uComp; // x unused, y particle gain, z unused, w unused

out vec4 fragColor;
void main()
{
	ivec2 p = ivec2(gl_FragCoord.xy);
	vec4 piece = texelFetch(sTD2DInputs[0], p, 0);
	vec4 vol = texelFetch(sTD2DInputs[1], p, 0);
	float mask = smoothstep(0.02, 0.08, max(piece.r, max(piece.g, piece.b)));
	vec3 rgb = mix(vol.rgb, piece.rgb, mask);
	fragColor = vec4(clamp(rgb, 0.0, 1.5), 1.0);
}

