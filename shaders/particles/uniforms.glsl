// Uniforms for init.glsl and the step include. Declared once; both stages include this before spawn.glsl.

uniform vec4 uWrite;      // x: 0 position+age, 1 velocity+life
uniform vec4 uPhys;       // drag (1/s), gravity, buoyancy per temperature, turbulence amplitude
uniform vec4 uTurb;       // x spatial frequency
uniform vec4 uField;      // x: 0 bus, 1 constant velocity, 2 xy rotation
uniform vec4 uAnalytic;   // constant vel xyz, or rotation centre xy + angular speed w
uniform vec4 uEmit;       // x mode, y field-accept threshold
uniform vec4 uEmitPos;    // xyz centre, w radius
uniform vec4 uEmitBox;    // xyz half extents
uniform vec4 uEmitVel;    // xyz base velocity, w random speed
uniform vec4 uLife;       // x min seconds, y max seconds
uniform vec4 uCollide;    // x radius (negative disables), y restitution, z friction, w gradient step
uniform vec4 uLimit;      // x max speed
uniform vec4 uBounds;     // 2D world xmin, ymin, xmax, ymax
uniform vec4 uBMin;
uniform vec4 uBMax;
uniform vec4 uSpawn;      // tracer position
uniform vec4 uSpawnVel;
uniform vec4 uClock;      // dt, time, step, frame
uniform vec4 uClock2;     // x seed
