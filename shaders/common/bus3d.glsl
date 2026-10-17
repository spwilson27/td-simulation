// 3D field bus contract (included with #include <bus3d_glsl>).
//
// World space: +y is up. One voxel per grid cell; voxel centres sit at the half-texel of vUV / busUV().
//   vel        rgba16float or rgba32float   xyz velocity in world units / s, w unused
//   scalars    rgba16float                  r smoke, g temperature, b fuel, a expansion rate (1/s)
//   pressure   mono32float                  pressure with dt folded in (v -= grad p)
//   sources    rgba32float                  per-second injection: r smoke, g heat, b fuel, a extra expansion
//   forces     rgba32float                  xyz acceleration in world units / s^2
//   obstacles  rgba32float                  r signed distance (world units, < 0 inside), gba obstacle velocity
//
// A side marked open is a Dirichlet pressure boundary (ghost pressure 0) and an absorbing band
// (busSponge). A closed side is a Neumann wall. All-Neumann pressure is singular once combustion
// adds divergence, so the open top is what lets that expansion leave.

uniform vec4 uBMin;     // xyz world min
uniform vec4 uBMax;     // xyz world max
uniform vec4 uWalls;    // closed (1) or open (0): -x, +x, -y, +y
uniform vec4 uWallsZ;   // closed (1) or open (0): -z, +z in xy
uniform vec4 uSponge;   // x band width (world), y fade rate (1/s)
uniform vec4 uClock;    // dt, time, step, frame
uniform vec4 uClock2;   // x seed

#define BUS_DT    (uClock.x)
#define BUS_TIME  (uClock.y)
#define BUS_SEED  (uClock2.x)
#define BUS_EMPTY_DISTANCE 1.0e4

vec3 busSize() { return uBMax.xyz - uBMin.xyz; }

vec3 busRes()
{
	return vec3(uTDOutputInfo.res.z, uTDOutputInfo.res.w, uTDOutputInfo.depth.y);
}

vec3 busTexel()
{
	return vec3(uTDOutputInfo.res.x, uTDOutputInfo.res.y, uTDOutputInfo.depth.x);
}

float busCell() { return busSize().x * uTDOutputInfo.res.x; }

vec3 busUVToWorld(vec3 uv) { return uBMin.xyz + uv * busSize(); }
vec3 busWorldToUV(vec3 p)  { return (p - uBMin.xyz) / busSize(); }

ivec3 busCoord() { return ivec3(gl_GlobalInvocationID.xyz); }

bool busLive(ivec3 c)
{
	return all(greaterThanEqual(c, ivec3(0))) && all(lessThan(c, ivec3(busRes())));
}

vec3 busUV(ivec3 c) { return (vec3(c) + 0.5) * busTexel(); }

bool busOutside(vec3 uv)
{
	return any(lessThan(uv, vec3(0.0))) || any(greaterThan(uv, vec3(1.0)));
}

vec3 busClampUV(vec3 uv)
{
	vec3 t = busTexel();
	return clamp(uv, 0.5 * t, 1.0 - 0.5 * t);
}

bool busSolid(vec4 obstacle) { return obstacle.r < 0.0; }
vec3 busObstacleVel(vec4 obstacle) { return obstacle.gba; }

// `dir` is the step from a cell to a neighbour. An open domain side is a Dirichlet pressure
// boundary (ghost pressure 0) so a net expansion has somewhere to leave. A closed side mirrors
// pressure (Neumann). Without at least one Dirichlet side the Poisson problem is singular:
// combustion adds divergence and the walls cannot carry it, so a residual floor is left behind.
bool busSideOpen(ivec3 dir)
{
	if (dir.x > 0) return uWalls.y < 0.5;
	if (dir.x < 0) return uWalls.x < 0.5;
	if (dir.y > 0) return uWalls.w < 0.5;
	if (dir.y < 0) return uWalls.z < 0.5;
	if (dir.z > 0) return uWallsZ.y < 0.5;
	if (dir.z < 0) return uWallsZ.x < 0.5;
	return false;
}

float busPressureGhost(ivec3 dir, float centre)
{
	return busSideOpen(dir) ? 0.0 : centre;
}

float busSponge(vec3 uv)
{
	if (uSponge.x <= 0.0)
		return 0.0;
	vec3 p = busUVToWorld(uv);
	float band = uSponge.x;
	float m = 0.0;
	vec3 dMin = p - uBMin.xyz;
	vec3 dMax = uBMax.xyz - p;
	m = max(m, clamp(1.0 - dMin.x / band, 0.0, 1.0) * (1.0 - step(0.5, uWalls.x)));
	m = max(m, clamp(1.0 - dMax.x / band, 0.0, 1.0) * (1.0 - step(0.5, uWalls.y)));
	m = max(m, clamp(1.0 - dMin.y / band, 0.0, 1.0) * (1.0 - step(0.5, uWalls.z)));
	m = max(m, clamp(1.0 - dMax.y / band, 0.0, 1.0) * (1.0 - step(0.5, uWalls.w)));
	m = max(m, clamp(1.0 - dMin.z / band, 0.0, 1.0) * (1.0 - step(0.5, uWallsZ.x)));
	m = max(m, clamp(1.0 - dMax.z / band, 0.0, 1.0) * (1.0 - step(0.5, uWallsZ.y)));
	return m * m;
}

vec3 busBacktrace(sampler3D vel, vec3 uv, float dt)
{
	vec3 size = busSize();
	vec3 v1 = texture(vel, uv).xyz;
	vec3 mid = busClampUV(uv - 0.5 * dt * v1 / size);
	vec3 v2 = texture(vel, mid).xyz;
	return busClampUV(uv - dt * v2 / size);
}

float sdSphere(vec3 p, vec3 c, float r) { return length(p - c) - r; }

float sdBox(vec3 p, vec3 c, vec3 halfSize)
{
	vec3 q = abs(p - c) - halfSize;
	return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r)
{
	vec3 pa = p - a, ba = b - a;
	float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0);
	return length(pa - ba * h) - r;
}

float busHash(vec3 p)
{
	p = fract(p * 0.1031);
	p += dot(p, p.yzx + 33.33);
	return fract((p.x + p.y) * p.z);
}

float busNoise(vec3 p)
{
	vec3 i = floor(p);
	vec3 f = fract(p);
	vec3 u = f * f * (3.0 - 2.0 * f);
	return mix(mix(mix(busHash(i), busHash(i + vec3(1, 0, 0)), u.x),
	               mix(busHash(i + vec3(0, 1, 0)), busHash(i + vec3(1, 1, 0)), u.x), u.y),
	           mix(mix(busHash(i + vec3(0, 0, 1)), busHash(i + vec3(1, 0, 1)), u.x),
	               mix(busHash(i + vec3(0, 1, 1)), busHash(i + vec3(1, 1, 1)), u.x), u.y), u.z);
}

float busFbm(vec3 p)
{
	float s = 0.0, a = 0.5;
	for (int i = 0; i < 4; i++) { s += a * busNoise(p); p = p * 2.03 + 1.7; a *= 0.5; }
	return s;
}
