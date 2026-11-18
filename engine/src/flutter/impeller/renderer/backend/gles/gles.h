// Copyright 2013 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef FLUTTER_IMPELLER_RENDERER_BACKEND_GLES_GLES_H_
#define FLUTTER_IMPELLER_RENDERER_BACKEND_GLES_GLES_H_

// IWYU pragma: begin_exports
// gl31.h is a superset of gl3.h. Its compute entry points and enums are only
// used when the context reports OpenGL ES 3.1 or OpenGL 4.3.
#include "GLES3/gl31.h"  // nogncheck

// Defines for extension enums.
#define IMPELLER_GL_CLAMP_TO_BORDER 0x812D
#define IMPELLER_GL_TEXTURE_BORDER_COLOR 0x1004

// GL_EXT_texture_filter_anisotropic
#define IMPELLER_GL_TEXTURE_MAX_ANISOTROPY 0x84FE
#define IMPELLER_GL_MAX_TEXTURE_MAX_ANISOTROPY 0x84FF

#define GL_GLEXT_PROTOTYPES
#include "GLES2/gl2ext.h"
// IWYU pragma: end_exports

#endif  // FLUTTER_IMPELLER_RENDERER_BACKEND_GLES_GLES_H_
