#include <uniforms_glsl>
#include <hash_glsl>
#include <spawn_glsl>
// 3D bus particle step. in0/in1 are the 2D particle textures. The bus volumes are sTD3DInputs
// in connection order: velocity, scalars, obstacles.

float particleDim() { return 1.0; }

vec3 toUV(vec3 p)
{
	return clamp((p - uBMin.xyz) / max(uBMax.xyz - uBMin.xyz, vec3(1e-6)), vec3(0.0), vec3(1.0));
}

vec3 sampleVel(vec3 p)
{
	return texture(sTD3DInputs[0], toUV(p)).xyz;
}

vec4 sampleScalars(vec3 p)
{
	return texture(sTD3DInputs[1], toUV(p));
}

vec4 sampleObs(vec3 p)
{
	vec4 tex = texture(sTD3DInputs[2], toUV(p));
	return vec4(tex.r, tex.gba);
}

#include <step_glsl>

out vec4 fragColor;
void main()
{
	fragColor = TDOutputSwizzle(particleOutput(ivec2(gl_FragCoord.xy)));
}
