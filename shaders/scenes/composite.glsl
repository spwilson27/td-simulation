// Combined scene. in0 volume, in1 pieces (near-black is empty), in2 liquid, in3 cloth (alpha).
// Debris is a second TOP (shaders/scenes/debris.glsl); this one already uses four inputs.
uniform vec4 uGain; // x piece, y liquid, z cloth, w unused

out vec4 fragColor;
void main()
{
	ivec2 p = ivec2(gl_FragCoord.xy);
	vec3 vol = texelFetch(sTD2DInputs[0], p, 0).rgb;
	vec3 piece = texelFetch(sTD2DInputs[1], p, 0).rgb;
	vec3 liquid = texelFetch(sTD2DInputs[2], p, 0).rgb;
	vec4 cloth = texelFetch(sTD2DInputs[3], p, 0);
	float pieceMask = smoothstep(0.02, 0.08, max(piece.r, max(piece.g, piece.b)));
	float liquidMask = smoothstep(0.04, 0.12, max(liquid.r, max(liquid.g, liquid.b)));
	vec3 rgb = vol;
	rgb = mix(rgb, liquid, liquidMask * uGain.y);
	rgb = mix(rgb, piece, pieceMask * uGain.x);
	rgb = mix(rgb, cloth.rgb, clamp(cloth.a, 0.0, 1.0) * uGain.z);
	fragColor = vec4(clamp(rgb, 0.0, 1.5), 1.0);
}
