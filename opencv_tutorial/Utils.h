#pragma once
#define NDEBUG
#include <assert.h>
//#define USE_RGB
#define USE_GRAY
//#define SHOW_COST
//#define SKIP_LOCAL
//#define SAVE_MESH
//#define DRAW_LSD
//#define DRAW_LINE
#define WRAPING_RESOLUTION 1e6


#define median(x,a,b)    (((a)<(b)) \
? ((x)<(a))?(a):(((x)>(b))?(b):(x)) \
: ((x)<(b))?(b):(((x)>(a))?(a):(x)) \
)
class Utils
{
};

