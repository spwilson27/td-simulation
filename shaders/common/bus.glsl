// Field bus contract shared by every simulation shader (included with #include <bus_glsl>).
//
// World space: +y is up, units are arbitrary but consistent (the demo uses metres).
// Fields (2D, one texel per grid cell, texel centres at vUV):
//   vel        rg32float    velocity in world units / s
//   scalars    rgba16float  r smoke density, g temperature, b fuel, a expansion rate (1/s, written by combustion)
//   pressure   mono32float  pressure (dt folded in: v -= grad p)
//   sources    rgba32float  per-second injection: r smoke, g heat, b fuel, a extra expansion (1/s)
//   forces     rgba32float  xy acceleration in world units / s^2
//   obstacles  rgba32float  r signed distance in world units (< 0 inside), gb obstacle velocity, a unused

uniform vec4 uBounds;   // world xmin, ymin, xmax, ymax
uniform vec4 uClock;    // dt (s), sim time (s), step, frame
uniform vec4 uClock2;   // x seed (SimClock uSeed). yzw reserved. Constant for a bake; splat noise is offset by it.
uniform vec4 uWalls;    // domain sides closed (1) or open (0): left, right, bottom, top
uniform vec4 uSponge;   // absorbing band at open sides: x width (world units), y fade rate (1/s)

#define BUS_DT    (uClock.x)
#define BUS_TIME  (uClock.y)
#define BUS_SEED  (uClock2.x)
#define BUS_EMPTY_DISTANCE 1.0e4

vec2 busSize()             { return uBounds.zw - uBounds.xy; }
vec2 busUVToWorld(vec2 uv) { return uBounds.xy + uv * busSize(); }
vec2 busWorldToUV(vec2 p)  { return (p - uBounds.xy) / busSize(); }
vec2 busTexel()            { return uTDOutputInfo.res.xy; }               // uv size of one cell
float busCell()            { return busSize().x * uTDOutputInfo.res.x; }  // world size of one cell (square cells)

bool busSolid(vec4 obstacle) { return obstacle.r < 0.0; }

// Domain sides are walls for the pressure solve (no flow through, pressure mirrored). A side marked open instead
// gets an absorbing band where velocity, smoke and heat fade out, so material appears to leave. A true open side
// (pressure 0 past it) went unstable on this collocated grid as soon as the plume reached it.
bool busOutside(vec2 uv) { return any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))); }
vec2 busClampUV(vec2 uv) { return clamp(uv, 0.5 * busTexel(), 1.0 - 0.5 * busTexel()); }

// 0 in the interior, rising to 1 at an open side.
float busSponge(vec2 uv)
{
	if (uSponge.x <= 0.0)
		return 0.0;
	vec2 p = busUVToWorld(uv);
	vec4 d = vec4(p.x - uBounds.x, uBounds.z - p.x, p.y - uBounds.y, uBounds.w - p.y);
	vec4 w = clamp(1.0 - d / uSponge.x, 0.0, 1.0) * (1.0 - step(0.5, uWalls));
	float m = max(max(w.x, w.y), max(w.z, w.w));
	return m * m;
}

// Semi-Lagrangian RK2 backtrace: where did the material at uv come from one step ago?
vec2 busBacktrace(sampler2D vel, vec2 uv, float dt)
{
	vec2 v1 = texture(vel, uv).xy;
	vec2 mid = busClampUV(uv - 0.5 * dt * v1 / busSize());
	vec2 v2 = texture(vel, mid).xy;
	return busClampUV(uv - dt * v2 / busSize());
}

float sdCircle(vec2 p, vec2 c, float r) { return length(p - c) - r; }
float sdBox(vec2 p, vec2 c, vec2 halfSize)
{
	vec2 d = abs(p - c) - halfSize;
	return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float busHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }
float busNoise(vec2 p)
{
	vec2 i = floor(p), f = fract(p);
	vec2 u = f * f * (3.0 - 2.0 * f);
	return mix(mix(busHash(i), busHash(i + vec2(1, 0)), u.x), mix(busHash(i + vec2(0, 1)), busHash(i + vec2(1, 1)), u.x), u.y);
}
float busFbm(vec2 p)
{
	float s = 0.0, a = 0.5;
	for (int i = 0; i < 4; i++) { s += a * busNoise(p); p *= 2.03; a *= 0.5; }
	return s;
}
