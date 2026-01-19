#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>

// Windows compatibility
#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#include <string.h>
#define snprintf _snprintf
#define strdup _strdup
#else
#include <string.h>
#endif

#endif // COMMON_H
