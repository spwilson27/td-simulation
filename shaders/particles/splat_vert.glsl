// td: stage vertex
// One point per particle. gl_VertexIndex matches toptoPOP's row-major order (y = 0 at the bottom).
// uMode.x > 0.5 scatters into a 3D slice atlas; otherwise into the 2D bus image.
// sReact is the deposit texture (xyz force, w heat).

uniform sampler2D sReact;
uniform sampler2D sState;
uniform vec4 uMode;
uniform vec4 uBounds;
uniform vec4 uBMin;
uniform vec4 uBMax;
uniform vec4 uAtlas;    // cols, rows, volume width, volume height
uniform vec4 uAtlas2;   // atlas width, atlas height, depth
uniform vec4 uRes;      // bus width, bus height

out vec4 vReact;

void main()
{
	ivec2 size = textureSize(sReact, 0);
	int id = TDInstanceID();
	ivec2 coord = ivec2(id % max(size.x, 1), id / max(size.x, 1));
	vReact = texelFetch(sReact, coord, 0);
	vec3 pos = texelFetch(sState, coord, 0).xyz;
	if (dot(abs(vReact), vec4(1.0)) < 1e-8)
	{
		gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
		gl_PointSize = 1.0;
		return;
	}
	vec2 pix;
	vec2 target;
	if (uMode.x > 0.5)
	{
		vec3 vol = vec3(uAtlas.z, uAtlas.w, uAtlas2.z);
		vec3 uvw = clamp((pos - uBMin.xyz) / max(uBMax.xyz - uBMin.xyz, vec3(1e-6)), vec3(0.0), vec3(1.0));
		ivec3 ijk = ivec3(clamp(floor(uvw * vol), vec3(0.0), vol - 1.0));
		int cols = max(int(uAtlas.x), 1);
		int col = ijk.z % cols;
		int row = ijk.z / cols;
		pix = vec2(col * int(vol.x) + ijk.x, row * int(vol.y) + ijk.y);
		target = uAtlas2.xy;
	}
	else
	{
		vec2 uv = clamp((pos.xy - uBounds.xy) / max(uBounds.zw - uBounds.xy, vec2(1e-6)), vec2(0.0), vec2(1.0));
		pix = clamp(floor(uv * uRes.xy), vec2(0.0), uRes.xy - 1.0);
		target = uRes.xy;
	}
	vec2 ndc = ((pix + 0.5) / max(target, vec2(1.0))) * 2.0 - 1.0;
	gl_Position = vec4(ndc, 0.0, 1.0);
	gl_PointSize = 1.0;
}
