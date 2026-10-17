// Geometric emitters. Field emission (fire / smoke) lives in the step shader, which can read the bus.
// Emitter menu: 0 point, 1 sphere, 2 box, 3 fire, 4 smoke, 5 tracer.

vec3 spherePoint(vec3 rnd, float radius)
{
	float z = rnd.y * 2.0 - 1.0;
	float a = rnd.z * 6.28318530718;
	float rxy = sqrt(max(0.0, 1.0 - z * z));
	vec3 direction = vec3(rxy * cos(a), rxy * sin(a), z);
	return direction * radius * pow(max(rnd.x, 1e-4), 1.0 / 3.0);
}

vec3 boxPoint(vec3 rnd, vec3 halfExt)
{
	return (rnd * 2.0 - 1.0) * halfExt;
}

void geometricSpawn(uint id, uint salt, out vec3 pos, out vec3 vel, out float life, out float age)
{
	float lifeTotal = mix(uLife.x, uLife.y, urand(id, salt, 1u));
	float phase = urand(id, salt, 9u);
	age = phase * lifeTotal;
	life = max(lifeTotal - age, 0.0);
	vec3 rnd = vec3(urand(id, salt, 2u), urand(id, salt, 3u), urand(id, salt, 4u));
	float mode = uEmit.x;
	if (mode < 0.5)
		pos = uEmitPos.xyz;
	else if (mode < 2.5 && mode >= 1.5)
		pos = uEmitPos.xyz + boxPoint(rnd, uEmitBox.xyz);
	else
		pos = uEmitPos.xyz + spherePoint(rnd, max(uEmitPos.w, 1e-4));
	vec3 rndv = vec3(urand(id, salt, 5u), urand(id, salt, 6u), urand(id, salt, 7u));
	vel = uEmitVel.xyz + (rndv * 2.0 - 1.0) * uEmitVel.w;
}
