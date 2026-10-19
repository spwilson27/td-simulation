uniform vec4 uBurst; // x > 0.5: inputs 2 and 3 are a one-step debris spawn (3D step only)

// Particle advance. The parent shader defines particleDim(), sampleVel(), sampleScalars(), sampleObs()
// before including this file. sampleObs() is (signed distance, velocity xyz).
//
// uWrite.x: 0 position.xyz + age, 1 velocity.xyz + life remaining.
// uField.x: 0 sample the bus, 1 constant velocity (uAnalytic.xyz), 2 solid-body rotation
//           about uAnalytic.xy with angular speed uAnalytic.w.
// Drag is integrated semi-implicitly; the remaining acceleration is an RK2 sample. With drag 0 and no
// extra acceleration the step is p += dt * v exactly, which is what the analytic path check uses.

vec3 domainLo()
{
	return particleDim() < 0.5 ? vec3(uBounds.xy, 0.0) : uBMin.xyz;
}

vec3 domainHi()
{
	return particleDim() < 0.5 ? vec3(uBounds.zw, 0.0) : uBMax.xyz;
}

vec3 fieldVelocity(vec3 p)
{
	if (uField.x > 1.5)
	{
		vec2 r = p.xy - uAnalytic.xy;
		return vec3(-uAnalytic.w * r.y, uAnalytic.w * r.x, 0.0);
	}
	if (uField.x > 0.5)
		return uAnalytic.xyz;
	return sampleVel(p);
}

vec3 turbulence(vec3 p)
{
	if (uPhys.w == 0.0)
		return vec3(0.0);
	vec3 q = p * uTurb.x + vec3(uClock.y * 0.15);
	float nx = hashNoise(q) * 2.0 - 1.0;
	float ny = hashNoise(q + vec3(19.1, 0.0, 0.0)) * 2.0 - 1.0;
	float nz = hashNoise(q + vec3(0.0, 19.1, 0.0)) * 2.0 - 1.0;
	return vec3(nx, ny, nz);
}

vec3 extraAccel(vec3 p)
{
	float temp = uField.x > 0.5 ? 0.0 : sampleScalars(p).g;
	return vec3(0.0, -uPhys.y + uPhys.z * temp, 0.0) + turbulence(p) * uPhys.w;
}

vec3 limitSpeed(vec3 v)
{
	float m = uLimit.x;
	float sp = length(v);
	return sp > m && m > 0.0 ? v * (m / sp) : v;
}

void confine(inout vec3 pos, inout vec3 vel)
{
	if (particleDim() < 0.5)
	{
		pos.z = 0.0;
		vel.z = 0.0;
	}
	vec3 lo = domainLo();
	vec3 hi = domainHi();
	vec3 nxt = clamp(pos, lo, hi);
	// Keep the velocity component where the clamp did nothing; clear it where the particle hit a side.
	vel *= vec3(1.0) - step(vec3(1e-8), abs(nxt - pos));
	pos = nxt;
}

void collide(inout vec3 pos, inout vec3 vel)
{
	if (uCollide.x < 0.0 || uField.x > 0.5)
		return;
	float radius = uCollide.x;
	float h = max(uCollide.w, 1e-4);
	for (int i = 0; i < 4; i++)
	{
		vec4 obs = sampleObs(pos);
		float sd = obs.x;
		if (sd >= radius)
			break;
		float dx = sampleObs(pos + vec3(h, 0.0, 0.0)).x - sampleObs(pos - vec3(h, 0.0, 0.0)).x;
		float dy = sampleObs(pos + vec3(0.0, h, 0.0)).x - sampleObs(pos - vec3(0.0, h, 0.0)).x;
		float dz = sampleObs(pos + vec3(0.0, 0.0, h)).x - sampleObs(pos - vec3(0.0, 0.0, h)).x;
		vec3 n = vec3(dx, dy, dz);
		float lenN = length(n);
		n = lenN > 1e-6 ? n / lenN : vec3(0.0, 1.0, 0.0);
		if (particleDim() < 0.5)
			n.z = 0.0;
		pos += n * (radius + 0.25 * h - sd);
		vec3 vobs = obs.yzw;
		vec3 rel = vel - vobs;
		float vn = dot(rel, n);
		if (vn < 0.0)
		{
			rel -= (1.0 + uCollide.y) * vn * n;
			float vn2 = dot(rel, n);
			vec3 vt = rel - vn2 * n;
			rel = vn2 * n + vt * (1.0 - uCollide.z);
			vel = vobs + rel;
		}
	}
}

vec3 domainPoint(uint id, uint salt, uint attempt)
{
	vec3 rnd = vec3(urand(id, salt, 20u + attempt), urand(id, salt, 21u + attempt), urand(id, salt, 22u + attempt));
	return mix(domainLo(), domainHi(), rnd);
}

// The plume sits above the emitter. Sampling the whole domain misses it.
vec3 plumePoint(uint id, uint salt, uint attempt)
{
	vec3 rnd = vec3(urand(id, salt, 30u + attempt), urand(id, salt, 31u + attempt), urand(id, salt, 32u + attempt));
	float r = max(uEmitPos.w, 0.05) * 3.0;
	vec3 p = uEmitPos.xyz;
	p.x += mix(-r, r, rnd.x);
	p.z += mix(-r, r, rnd.z);
	p.y = mix(uEmitPos.y, min(uEmitPos.y + 1.2, domainHi().y), rnd.y);
	return clamp(p, domainLo(), domainHi());
}

void fieldSpawn(uint id, uint salt, inout vec3 pos, inout vec3 vel, inout float life, inout float age)
{
	// A fraction of the dead pool is born each step, so the plume fills in instead of bursting.
	float dt = max(uClock.x, 1e-4);
	float meanLife = max(0.5 * (uLife.x + uLife.y), dt);
	if (urand(id, salt, 11u) > min(dt / meanLife, 1.0))
	{
		life = 0.0;
		age = 0.0;
		return;
	}
	bool fire = uEmit.x < 3.5;
	vec3 best = pos;
	float bestW = -1.0;
	for (uint k = 0u; k < 8u; k++)
	{
		vec3 candidate = k < 6u ? plumePoint(id, salt, k) : domainPoint(id, salt, k);
		if (particleDim() < 0.5)
			candidate.z = 0.0;
		vec4 s = sampleScalars(candidate);
		float w = fire ? s.g * max(s.b, 0.15) : s.r;
		if (w > bestW)
		{
			bestW = w;
			best = candidate;
		}
	}
	if (bestW < uEmit.y)
	{
		life = 0.0;
		age = 0.0;
		return;
	}
	pos = best;
	age = 0.0;
	life = mix(uLife.x, uLife.y, urand(id, salt, 1u));
	vec3 rndv = vec3(urand(id, salt, 5u), urand(id, salt, 6u), urand(id, salt, 7u));
	vel = fieldVelocity(pos) + (rndv * 2.0 - 1.0) * uEmitVel.w;
	if (particleDim() < 0.5)
		vel.z = 0.0;
}

vec4 particleOutput(ivec2 coord)
{
	ivec2 size = ivec2(uTDOutputInfo.res.zw);
	uint id = uint(coord.x) + uint(coord.y) * uint(size.x);
	vec4 posTex = texelFetch(sTD2DInputs[0], coord, 0);
	vec4 velTex = texelFetch(sTD2DInputs[1], coord, 0);
	vec3 pos = posTex.xyz;
	vec3 vel = velTex.xyz;
	float age = posTex.w;
	float life = velTex.w;
	bool tracer = uEmit.x > 4.5;
	bool fromBurst = false;
	// Only the 3D step sets uBurst. Its inputs 2 and 3 are the burst textures, wired ahead of the
	// bus volumes so they cannot be confused with a velocity field.
	if (life <= 0.0 && uBurst.x > 0.5)
	{
		vec4 burstP = texelFetch(sTD2DInputs[2], coord, 0);
		if (burstP.w > 0.0)
		{
			vec4 burstV = texelFetch(sTD2DInputs[3], coord, 0);
			pos = burstP.xyz;
			vel = burstV.xyz;
			age = 0.0;
			life = burstP.w;
			fromBurst = true;
		}
	}
	if (life <= 0.0 && !fromBurst)
	{
		if (tracer)
			return uWrite.x < 0.5 ? vec4(pos, age) : vec4(vel, life);
		uint salt = uint(uClock2.x) * 131u + uint(uClock.z) * 17u + 3u;
		if (uEmit.x > 2.5 && uEmit.x < 4.5 && uField.x < 0.5)
			fieldSpawn(id, salt, pos, vel, life, age);
		else
		{
			geometricSpawn(id, salt, pos, vel, life, age);
			age = 0.0;
			life = mix(uLife.x, uLife.y, urand(id, salt, 1u));
		}
	}
	float dt = uClock.x;
	float drag = max(uPhys.x, 0.0);
	vec3 vf1 = fieldVelocity(pos);
	vec3 a1 = extraAccel(pos);
	float halfDt = 0.5 * dt;
	vec3 v1 = (vel + halfDt * (drag * vf1 + a1)) / (1.0 + halfDt * drag);
	vec3 mid = pos + halfDt * v1;
	vec3 vf2 = fieldVelocity(mid);
	vec3 a2 = extraAccel(mid);
	vec3 v2 = (vel + dt * (drag * vf2 + a2)) / (1.0 + dt * drag);
	vec3 p2 = pos + dt * v2;
	v2 = limitSpeed(v2);
	collide(p2, v2);
	confine(p2, v2);
	age += dt;
	life = max(life - dt, 0.0);
	return uWrite.x < 0.5 ? vec4(p2, age) : vec4(v2, life);
}
