/*
 * This is a Gau2Grid automatically generated C file.
 *
 * More details can found at the following repo:
 *   https://github.com/dgasmith/gau2grid
 */



// ISOC11 does not seem to be well implemented across platforms and compilers
// This is a collection of macros to change pragmas and function calls as needed for compat.

#pragma once


#if defined(__GG_NO_PRAGMA)
    // Turn everything off if there are issues

    #define ALIGNED_MALLOC(alignment, size)                  malloc(size)
    #define ALIGNED_FREE(ptr)                                free(ptr)
    #define ASSUME_ALIGNED(ptr, width)

    #define PRAGMA_VECTORIZE
    #define PRAGMA_RESTRICT

#elif defined(__ICC) || defined(__INTEL_COMPILER)
    // pragmas for Intel

    #define ALIGNED_MALLOC(alignment, size)                  _mm_malloc(size, alignment)
    #define ALIGNED_FREE(ptr)                                _mm_free(ptr)
    #define ASSUME_ALIGNED(ptr, width)                       __assume_aligned(ptr, width)

    #define PRAGMA_VECTORIZE                                 _Pragma("vector")
    #define PRAGMA_RESTRICT                                  __restrict__

#elif defined(__clang__) && defined(_MSC_VER)
    // pragmas for MSVC

    #define ALIGNED_MALLOC(alignment, size)                  _aligned_malloc(size, alignment)
    #define ALIGNED_FREE(ptr)                                _aligned_free(ptr)
    #define ASSUME_ALIGNED(ptr, width)

    #define PRAGMA_VECTORIZE                                 __pragma(loop(ivdep))
    #define PRAGMA_RESTRICT                                  __restrict

#elif defined(__clang__)
    // pragmas for Clang.
    // Do this before GCC because clang also defines __GNUC__

    #define ALIGNED_MALLOC(alignment, size)                  _mm_malloc(size, alignment)
    #define ALIGNED_FREE(ptr)                                _mm_free(ptr)
    #define ASSUME_ALIGNED(ptr, width)

    #define PRAGMA_VECTORIZE                                 _Pragma("clang loop vectorize(enable)")
    #define PRAGMA_RESTRICT                                  __restrict__

#elif (defined(__GNUC__) || defined(__GNUG__)) && defined(__APPLE__)
    // pragmas for GCC on Darwin (weird aligned alloc not found on Darwin)

    #define ALIGNED_MALLOC(alignment, size)                  malloc(size)
    #define ALIGNED_FREE(ptr)                                free(ptr)
    #define ASSUME_ALIGNED(ptr, width)

    #define PRAGMA_VECTORIZE                                 _Pragma("GCC ivdep")
    #define PRAGMA_RESTRICT                                  __restrict__

#elif defined(__GNUC__) || defined(__GNUG__)
    // pragmas for GCC

    #define ALIGNED_MALLOC(alignment, size)                  aligned_alloc(alignment, size)
    #define ALIGNED_FREE(ptr)                                free(ptr)
    #define ASSUME_ALIGNED(ptr, width)

    #define PRAGMA_VECTORIZE                                 _Pragma("GCC ivdep")
    #define PRAGMA_RESTRICT                                  __restrict__

#elif defined(_MSC_VER)
    // pragmas for MSVC

    #define ALIGNED_MALLOC(alignment, size)                  _aligned_malloc(size, alignment)
    #define ALIGNED_FREE(ptr)                                _aligned_free(ptr)
    #define ASSUME_ALIGNED(ptr, width)

    #define PRAGMA_VECTORIZE                                 __pragma(loop(ivdep))
    #define PRAGMA_RESTRICT                                  __restrict


#elif defined(__PGI)
    // pragmas for PGI

    #define ALIGNED_MALLOC(alignment, size)                  aligned_alloc(alignment, size)
    #define ALIGNED_FREE(ptr)                                free(ptr)
    #define ASSUME_ALIGNED(ptr, width)

    #define PRAGMA_VECTORIZE                                 _Pragma("ivdep")
    #define PRAGMA_RESTRICT                                  __restrict__


#endif