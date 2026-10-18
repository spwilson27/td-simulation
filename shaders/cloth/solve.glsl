#include <cloth_common_glsl>
// One substep, unrolled in passes. Pass 0 integrates. Passes 1..8 are stretch and shear.
// Passes 9..12 repeat stretch. Passes 13..16 are isometric bending. Passes 17..18 are
// long-range attachments. Pass 19 is ground and SDF. Pass 20 is self-collision.
// Bending is before the push so the sphere constraint is what the vertices keep.
// Input 0 is replaced each pass. Input 3 stays on the position at the start of the substep.
// 2D: pos, vel, rest, posStart, keys. 3D: wind, obstacles, bins.
uniform vec4 uDyn;    // dt, damping, max speed, collide
uniform vec4 uGrav;   // gravity xyz, air density
uniform vec4 uAero;   // drag, wind cap, friction, thickness
uniform vec4 uComp;   // stretch, shear, bend
uniform vec4 uLen;    // dx, dy
uniform vec4 uGround; // y
uniform vec4 uHash;   // resolution
uniform vec4 uAttach; // xyz offset, w yaw
uniform vec4 uOrigin;
uniform vec4 uPin; // x menu: 0 left, 1 corners, 2 none, 3 top
uniform vec4 uBMin;
uniform vec4 uBMax;
layout(local_size_x = 8, local_size_y = 8) in;

vec3 obsGrad(vec3 uv)
{
	vec3 t = 1.0 / vec3(textureSize(sTD3DInputs[1], 0));
	float dx = texture(sTD3DInputs[1], clamp(uv + vec3(t.x, 0.0, 0.0), 0.0, 1.0)).r
	         - texture(sTD3DInputs[1], clamp(uv - vec3(t.x, 0.0, 0.0), 0.0, 1.0)).r;
	float dy = texture(sTD3DInputs[1], clamp(uv + vec3(0.0, t.y, 0.0), 0.0, 1.0)).r
	         - texture(sTD3DInputs[1], clamp(uv - vec3(0.0, t.y, 0.0), 0.0, 1.0)).r;
	float dz = texture(sTD3DInputs[1], clamp(uv + vec3(0.0, 0.0, t.z), 0.0, 1.0)).r
	         - texture(sTD3DInputs[1], clamp(uv - vec3(0.0, 0.0, t.z), 0.0, 1.0)).r;
	return vec3(dx, dy, dz);
}

void role(int kind, int color, int i, int j, int nx, int ny, out int sgn, out int i0, out int j0, out int live)
{
	sgn = 0;
	i0 = 0;
	j0 = 0;
	live = 0;
	if (kind == 0)
	{
		if ((i & 1) == color && i + 1 < nx) { sgn = 1; i0 = i; j0 = j; live = 1; }
		else if (i > 0 && ((i - 1) & 1) == color) { sgn = -1; i0 = i - 1; j0 = j; live = 1; }
	}
	else if (kind == 1)
	{
		if ((j & 1) == color && j + 1 < ny) { sgn = 1; i0 = i; j0 = j; live = 1; }
		else if (j > 0 && ((j - 1) & 1) == color) { sgn = -1; i0 = i; j0 = j - 1; live = 1; }
	}
	else if (kind == 2)
	{
		if ((i & 1) == color && i + 1 < nx && j + 1 < ny) { sgn = 1; i0 = i; j0 = j; live = 1; }
		else if (i > 0 && j > 0 && ((i - 1) & 1) == color) { sgn = -1; i0 = i - 1; j0 = j - 1; live = 1; }
	}
	else if (kind == 3)
	{
		if (j > 0 && (i & 1) == color && i + 1 < nx) { sgn = 1; i0 = i; j0 = j - 1; live = 1; }
		else if (i > 0 && ((i - 1) & 1) == color && j + 1 < ny) { sgn = -1; i0 = i - 1; j0 = j; live = 1; }
	}
	else if (kind == 4)
	{
		if (((i / 2) & 1) == color && i + 2 < nx) { sgn = 1; i0 = i; j0 = j; live = 1; }
		else if (i >= 2 && (((i - 2) / 2) & 1) == color) { sgn = -1; i0 = i - 2; j0 = j; live = 1; }
	}
	else if (kind == 5)
	{
		if (((j / 2) & 1) == color && j + 2 < ny) { sgn = 1; i0 = i; j0 = j; live = 1; }
		else if (j >= 2 && (((j - 2) / 2) & 1) == color) { sgn = -1; i0 = i; j0 = j - 2; live = 1; }
	}
}

void endpoints(int kind, int i0, int j0, out ivec2 a, out ivec2 b)
{
	a = ivec2(i0, j0);
	b = ivec2(i0, j0);
	if (kind == 0) b = ivec2(i0 + 1, j0);
	else if (kind == 1) b = ivec2(i0, j0 + 1);
	else if (kind == 2) b = ivec2(i0 + 1, j0 + 1);
	else if (kind == 3) { a = ivec2(i0, j0 + 1); b = ivec2(i0 + 1, j0); }
	else if (kind == 4) b = ivec2(i0 + 2, j0);
	else if (kind == 5) b = ivec2(i0, j0 + 2);
}

float restOf(int kind)
{
	float diag = length(uLen.xy);
	if (kind == 1) return uLen.y;
	if (kind == 2 || kind == 3) return diag;
	if (kind == 4) return 2.0 * uLen.x;
	if (kind == 5) return 2.0 * uLen.y;
	return uLen.x;
}

float compOf(int kind)
{
	if (kind == 2 || kind == 3) return uComp.y;
	if (kind >= 4) return uComp.z;
	return uComp.x;
}

vec3 pinWorld(ivec2 at)
{
	vec3 restp = texelFetch(sTD2DInputs[2], at, 0).xyz;
	vec3 r = restp - uOrigin.xyz;
	float c = cos(uAttach.w);
	float s = sin(uAttach.w);
	return vec3(c * r.x + s * r.z, r.y, -s * r.x + c * r.z) + uOrigin.xyz + uAttach.xyz;
}

void projectPin(inout vec3 p, vec3 anchor, float restLen)
{
	if (restLen <= 1e-6)
		return;
	vec3 d = p - anchor;
	float dist = length(d);
	if (dist > restLen)
		p = anchor + d * (restLen / dist);
}

void main()
{
	ivec2 size = ivec2(uTDOutputInfo.res.zw);
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	if (coord.x >= size.x || coord.y >= size.y)
		return;
	vec4 cur = texelFetch(sTD2DInputs[0], coord, 0);
	vec3 p = cur.xyz;
	float inv = cur.w;
	float dt = max(uDyn.x, 1e-6);
	int pass = uTDPass;

	if (pass == 0)
	{
		if (inv <= 0.0)
		{
			vec3 restp = texelFetch(sTD2DInputs[2], coord, 0).xyz;
			vec3 r = restp - uOrigin.xyz;
			float c = cos(uAttach.w);
			float s = sin(uAttach.w);
			p = vec3(c * r.x + s * r.z, r.y, -s * r.x + c * r.z) + uOrigin.xyz + uAttach.xyz;
		}
		else
		{
			vec3 v = texelFetch(sTD2DInputs[1], coord, 0).xyz;
			v *= exp(-uDyn.y * dt);
			v += uGrav.xyz * dt;
			if (uGrav.w > 0.0)
			{
				vec3 span = max(uBMax.xyz - uBMin.xyz, vec3(1e-6));
				vec3 uv = clamp((p - uBMin.xyz) / span, vec3(0.0), vec3(1.0));
				vec3 n = clothNormal(sTD2DInputs[0], coord, size);
				vec3 force = clothAero(n, v, texture(sTD3DInputs[0], uv).xyz, uLen.x * uLen.y,
					uGrav.w, uAero.x, uAero.y);
				v += force * inv * dt;
			}
			p += v * dt;
		}
	}
	else if ((pass >= 1 && pass <= 8) || (pass >= 9 && pass <= 12))
	{
		int kind;
		int color;
		if (pass <= 8)
		{
			kind = (pass - 1) / 2;
			color = (pass - 1) - kind * 2;
		}
		else
		{
			int within = pass - 9;
			kind = within / 2;
			color = within - kind * 2;
		}
		int sgn, i0, j0, live;
		role(kind, color, coord.x, coord.y, size.x, size.y, sgn, i0, j0, live);
		if (live == 1)
		{
			float bits = texelFetch(sTD2DInputs[1], ivec2(i0, j0), 0).w;
			int broken = int(bits + 0.5);
			bool torn = (kind == 0 && (broken & 1) != 0) || (kind == 1 && (broken & 2) != 0);
			if (!torn)
			{
				ivec2 a, b;
				endpoints(kind, i0, j0, a, b);
				vec4 pa = texelFetch(sTD2DInputs[0], a, 0);
				vec4 pb = texelFetch(sTD2DInputs[0], b, 0);
				vec3 diff = pa.xyz - pb.xyz;
				float len = length(diff);
				float rest = restOf(kind);
				float strain = rest > 1e-8 ? (len - rest) / rest : 0.0;
				bool over = (kind <= 1) && uComp.w > 0.0 && strain > uComp.w;
				if (!over)
				{
					vec3 grad = len > 1e-8 ? diff / len : vec3(1.0, 0.0, 0.0);
					float alpha = compOf(kind) / (dt * dt);
					float denom = pa.w + pb.w + alpha;
					if (denom > 0.0)
					{
						float dl = -(len - rest) / denom;
						float wi = sgn > 0 ? pa.w : pb.w;
						p += float(sgn) * wi * dl * grad;
					}
				}
			}
		}
	}
	else if (pass >= 13 && pass <= 16 && inv > 0.0 && coord.x > 0 && coord.y > 0
		&& coord.x + 1 < size.x && coord.y + 1 < size.y)
	{
		// Isometric bending, after collisions. Gradients are constant, so the centre takes
		// only its share of a Jacobi step and neighbouring centres do not write this vertex.
		vec3 left = texelFetch(sTD2DInputs[0], coord + ivec2(-1, 0), 0).xyz;
		vec3 right = texelFetch(sTD2DInputs[0], coord + ivec2(1, 0), 0).xyz;
		vec3 down = texelFetch(sTD2DInputs[0], coord + ivec2(0, -1), 0).xyz;
		vec3 up = texelFetch(sTD2DInputs[0], coord + ivec2(0, 1), 0).xyz;
		float wl = texelFetch(sTD2DInputs[0], coord + ivec2(-1, 0), 0).w;
		float wr = texelFetch(sTD2DInputs[0], coord + ivec2(1, 0), 0).w;
		float wd = texelFetch(sTD2DInputs[0], coord + ivec2(0, -1), 0).w;
		float wu = texelFetch(sTD2DInputs[0], coord + ivec2(0, 1), 0).w;
		vec3 bend = 4.0 * p - left - right - down - up;
		float alpha = uComp.z / (dt * dt);
		float denom = 16.0 * inv + wl + wr + wd + wu + alpha;
		if (denom > 0.0)
			p += (-4.0 * inv / denom) * bend;
	}
	else if (pass >= 17 && pass <= 18 && inv > 0.0)
	{
		int mode = int(uPin.x + 0.5);
		int i = coord.x;
		int j = coord.y;
		if (mode == 3)
			projectPin(p, pinWorld(ivec2(i, 0)), float(j) * uLen.y);
		else if (mode == 0)
			projectPin(p, pinWorld(ivec2(0, j)), float(i) * uLen.x);
		else if (mode == 1)
		{
			projectPin(p, pinWorld(ivec2(0, 0)), float(i) * uLen.x + float(j) * uLen.y);
			projectPin(p, pinWorld(ivec2(size.x - 1, 0)), float(size.x - 1 - i) * uLen.x + float(j) * uLen.y);
		}
	}
	else if (pass == 19 && uDyn.w > 0.5 && inv > 0.0)
	{
		vec3 start = texelFetch(sTD2DInputs[3], coord, 0).xyz;
		float thick = max(uAero.w, 0.0);
		float friction = clamp(uAero.z, 0.0, 1.0);
		if (p.y < uGround.x + thick)
		{
			vec3 n = vec3(0.0, 1.0, 0.0);
			p.y = uGround.x + thick;
			vec3 rel = p - start;
			float nr = dot(rel, n);
			if (nr < 0.0)
				rel -= nr * n;
			vec3 tang = rel - dot(rel, n) * n;
			rel -= tang * friction;
			p = start + rel;
			p.y = max(p.y, uGround.x + thick);
		}
		vec3 span = max(uBMax.xyz - uBMin.xyz, vec3(1e-6));
		vec3 uv = clamp((p - uBMin.xyz) / span, vec3(0.0), vec3(1.0));
		vec4 obs = texture(sTD3DInputs[1], uv);
		if (obs.r < thick)
		{
			vec3 g = obsGrad(uv);
			float glen = length(g);
			vec3 n = glen > 1e-6 ? g / glen : vec3(0.0, 1.0, 0.0);
			p += n * (thick - obs.r);
			vec3 rel = (p - start) - obs.gba * dt;
			float nr = dot(rel, n);
			if (nr < 0.0)
				rel -= nr * n;
			vec3 tang = rel - dot(rel, n) * n;
			rel -= tang * friction;
			p = start + rel + obs.gba * dt;
			uv = clamp((p - uBMin.xyz) / span, vec3(0.0), vec3(1.0));
			obs = texture(sTD3DInputs[1], uv);
			if (obs.r < thick)
				p += n * (thick - obs.r);
		}
	}
	else if (pass == 20)
	{
		int id = coord.x + coord.y * size.x;
		p = clothSelf(p, inv, id, size, sTD2DInputs[0], sTD2DInputs[4], sTD3DInputs[2],
			uBMin.xyz, uBMax.xyz, max(int(uHash.x + 0.5), 1), uAero.w);
	}
	TDImageStoreOutput(0, ivec3(coord, 0), vec4(p, inv));
}
