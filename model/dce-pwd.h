#ifndef SIMU_PWD_H
#define SIMU_PWD_H

#include <sys/types.h>
#include <pwd.h>

#ifdef __cplusplus
extern "C" {
#endif

struct passwd * dce_getpwnam (const char *name);

struct passwd * dce_getpwuid (uid_t uid);

void dce_endpwent (void);

int dce_getpwnam_r (const char *name, struct passwd *pwd, char *buf, size_t buflen, struct passwd **result);
int dce_getpwuid_r (uid_t uid, struct passwd *pwd, char *buf, size_t buflen, struct passwd **result);

#ifdef __cplusplus
}
#endif

#endif /* SIMU_PWD_H */
