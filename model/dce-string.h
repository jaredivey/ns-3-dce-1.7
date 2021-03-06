#ifndef SIMU_STRING_H
#define SIMU_STRING_H

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

char * dce_strdup (const char *s);
char * dce_strndup (const char *s, size_t n);
char * dce___strcpy_chk (char *__restrict __dest,
                         const char *__restrict __src,
                         size_t __destlen);
char * dce_strpbrk (const char *s, const char *accept);
char * dce_strstr (const char *h, const char *n);
void * dce_memcpy (void *dest, const void *source, size_t num);
void * dce___rawmemchr (const void *s, int c);
void * dce___memcpy_chk(void * dest, const void * src, size_t len, size_t destlen);

#ifdef __cplusplus
}
#endif

#endif /* SIMU_STRING_H */
