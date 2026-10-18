#include <bus3d_glsl>
#include <liquid_common_glsl>
// Add the coarse correction onto the fine pressure. Non-liquid cells stay at 0.
// in0 fine pressure, in1 coarse correction, in2 fine type.
// td: output3d
uniform vec4 uScale;
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

void main()
{
	ivec3 c = busCoord();
	if (!busLive(c))
		return;
	if (lqKind(texelFetch(sTD3DInputs[2], c, 0).r) != LQ_LIQUID)
	{
		TDImageStoreOutput(0, c, vec4(0.0));
		return;
	}
	// Manual trilinear with clamp-to-edge. Hardware linear filtering blends the boundary
	// texel with a zero border, which lowers the floor pressure and points the hydrostatic
	// gradient the wrong way on the cells that carry the whole residual.
	ivec3 nc = textureSize(sTD3DInputs[1], 0);
	ivec3 fine = ivec3(busRes());
	vec3 idx = (vec3(c) + 0.5) * vec3(nc) / vec3(fine) - 0.5;
	ivec3 i0 = ivec3(floor(idx));
	vec3 t = idx - vec3(i0);
	ivec3 i1 = i0 + ivec3(1);
	i0 = clamp(i0, ivec3(0), nc - 1);
	i1 = clamp(i1, ivec3(0), nc - 1);
	float c000 = texelFetch(sTD3DInputs[1], ivec3(i0.x, i0.y, i0.z), 0).r;
	float c100 = texelFetch(sTD3DInputs[1], ivec3(i1.x, i0.y, i0.z), 0).r;
	float c010 = texelFetch(sTD3DInputs[1], ivec3(i0.x, i1.y, i0.z), 0).r;
	float c110 = texelFetch(sTD3DInputs[1], ivec3(i1.x, i1.y, i0.z), 0).r;
	float c001 = texelFetch(sTD3DInputs[1], ivec3(i0.x, i0.y, i1.z), 0).r;
	float c101 = texelFetch(sTD3DInputs[1], ivec3(i1.x, i0.y, i1.z), 0).r;
	float c011 = texelFetch(sTD3DInputs[1], ivec3(i0.x, i1.y, i1.z), 0).r;
	float c111 = texelFetch(sTD3DInputs[1], ivec3(i1.x, i1.y, i1.z), 0).r;
	float c00 = mix(c000, c100, t.x);
	float c01 = mix(c001, c101, t.x);
	float c10 = mix(c010, c110, t.x);
	float c11 = mix(c011, c111, t.x);
	float c0 = mix(c00, c10, t.y);
	float c1 = mix(c01, c11, t.y);
	float corr = mix(c0, c1, t.z) * uScale.x;
	float p = texelFetch(sTD3DInputs[0], c, 0).r + corr;
	TDImageStoreOutput(0, c, vec4(p, 0.0, 0.0, 1.0));
}
