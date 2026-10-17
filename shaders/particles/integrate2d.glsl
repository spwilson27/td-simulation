#include <uniforms_glsl>
#include <hash_glsl>
#include <spawn_glsl>
// 2D bus particle step. in0 previous position, in1 previous velocity, in2 bus velocity,
// in3 bus scalars, in4 bus obstacles.

float particleDim() { return 0.0; }

vec2 toUV(vec3 p)
{
	return clamp((p.xy - uBounds.xy) / max(uBounds.zw - uBounds.xy, vec2(1e-6)), vec2(0.0), vec2(1.0));
}

ivec2 cellAt(int index, vec3 p)
{
	vec2 uv = toUV(p);
	ivec2 size = textureSize(sTD2DInputs[index], 0);
	return clamp(ivec2(uv * vec2(size)), ivec2(0), size - ivec2(1));
}

vec3 sampleVel(vec3 p)
{
	return vec3(texelFetch(sTD2DInputs[2], cellAt(2, p), 0).xy, 0.0);
}

vec4 sampleScalars(vec3 p)
{
	return texelFetch(sTD2DInputs[3], cellAt(3, p), 0);
}

vec4 sampleObs(vec3 p)
{
	vec4 tex = texelFetch(sTD2DInputs[4], cellAt(4, p), 0);
	return vec4(tex.r, tex.g, tex.b, 0.0);
}

#include <step_glsl>

out vec4 fragColor;
void main()
{
	fragColor = TDOutputSwizzle(particleOutput(ivec2(gl_FragCoord.xy)));
}
