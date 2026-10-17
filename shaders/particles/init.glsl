#include <uniforms_glsl>
#include <hash_glsl>
#include <spawn_glsl>
// One texel per particle. uWrite.x < 0.5 writes position.xyz + age, otherwise velocity.xyz + life remaining.
// Input 0 is a dummy so the GLSL TOP has a wired input.

out vec4 fragColor;

void main()
{
	ivec2 size = ivec2(uTDOutputInfo.res.zw);
	ivec2 coord = ivec2(gl_FragCoord.xy);
	uint id = uint(coord.x) + uint(coord.y) * uint(size.x);
	uint salt = uint(uClock2.x);
	vec3 pos, vel;
	float life, age;
	if (uEmit.x > 4.5)
	{
		if (id == 0u)
		{
			pos = uSpawn.xyz;
			vel = uSpawnVel.xyz;
			age = 0.0;
			life = 1.0e6;
		}
		else
		{
			pos = vec3(0.0);
			vel = vec3(0.0);
			age = 0.0;
			life = 0.0;
		}
	}
	else if (uEmit.x > 2.5 && uEmit.x < 4.5)
	{
		// Fire and smoke start dead and are born across later steps from hot cells.
		pos = vec3(0.0);
		vel = vec3(0.0);
		age = 0.0;
		life = 0.0;
	}
	else
		geometricSpawn(id, salt, pos, vel, life, age);
	vec4 outPos = vec4(pos, age);
	vec4 outVel = vec4(vel, life);
	fragColor = TDOutputSwizzle(uWrite.x < 0.5 ? outPos : outVel);
}
