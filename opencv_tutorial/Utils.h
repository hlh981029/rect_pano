#pragma once
//#define USE_RGB
#define USE_GRAY
//#define SHOW_COST
#define SKIP_LOCAL
//#define SAVE_MESH
//#define DRAW_LSD
#define DRAW_LINE
#define median(x,a,b)    (((a)<(b)) \
? ((x)<(a))?(a):(((x)>(b))?(b):(x)) \
: ((x)<(b))?(b):(((x)>(a))?(a):(x)) \
)
class Utils
{
};

