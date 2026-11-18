// Copyright 2013 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Reads each vertex position from a storage buffer instead of a vertex
// buffer, so a compute shader's output can be drawn without a vertex layout.

uniform VertInfo {
  mat4 mvp;
  vec4 color;
}
vert_info;

layout(std430) readonly buffer Positions {
  vec2 positions[];
}
positions;

out vec4 v_color;

void main() {
  v_color = vert_info.color;
  gl_Position =
      vert_info.mvp * vec4(positions.positions[gl_VertexIndex], 0.0, 1.0);
}
