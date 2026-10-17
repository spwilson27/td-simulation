// Deterministic hashes shared by the particle init and step shaders.
// The integer mix matches particles_math.pcg / urand in python/particles_math.py.

uint pcg(uint v)
{
	uint state = v * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

float urand(uint x, uint y, uint salt)
{
	uint h = pcg(x + pcg(y + pcg(salt)));
	return float(h) * (1.0 / 4294967296.0);
}

float hashNoise(vec3 p)
{
	vec3 i = floor(p);
	vec3 f = fract(p);
	f = f * f * (3.0 - 2.0 * f);
	float n000 = urand(uint(int(i.x) * 17 + 4096), uint(int(i.y) * 17 + 4096), uint(int(i.z) * 17 + 8192));
	float n100 = urand(uint(int(i.x + 1.0) * 17 + 4096), uint(int(i.y) * 17 + 4096), uint(int(i.z) * 17 + 8192));
	float n010 = urand(uint(int(i.x) * 17 + 4096), uint(int(i.y + 1.0) * 17 + 4096), uint(int(i.z) * 17 + 8192));
	float n110 = urand(uint(int(i.x + 1.0) * 17 + 4096), uint(int(i.y + 1.0) * 17 + 4096), uint(int(i.z) * 17 + 8192));
	float n001 = urand(uint(int(i.x) * 17 + 4096), uint(int(i.y) * 17 + 4096), uint(int(i.z + 1.0) * 17 + 8192));
	float n101 = urand(uint(int(i.x + 1.0) * 17 + 4096), uint(int(i.y) * 17 + 4096), uint(int(i.z + 1.0) * 17 + 8192));
	float n011 = urand(uint(int(i.x) * 17 + 4096), uint(int(i.y + 1.0) * 17 + 4096), uint(int(i.z + 1.0) * 17 + 8192));
	float n111 = urand(uint(int(i.x + 1.0) * 17 + 4096), uint(int(i.y + 1.0) * 17 + 4096), uint(int(i.z + 1.0) * 17 + 8192));
	float nx00 = mix(n000, n100, f.x);
	float nx10 = mix(n010, n110, f.x);
	float nx01 = mix(n001, n101, f.x);
	float nx11 = mix(n011, n111, f.x);
	return mix(mix(nx00, nx10, f.y), mix(nx01, nx11, f.y), f.z);
}
