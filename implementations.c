#define crt_implementation
#include "crt.h"

#define quick_implementation
#include "quick.h"

#define FCL_IMPLEMENTATION
#include "fcl.h"

#pragma warning(push)
#pragma warning(disable: 4459) // parameter/local hides global declaration
#pragma warning(disable: 4244) // conversion from '...' to '...', possible loss of data
#define STBI_ASSERT(x) assert(x)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#pragma warning(pop)

