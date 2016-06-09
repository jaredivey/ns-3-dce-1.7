#ifndef PTHREAD_FIBER_MANAGER_H
#define PTHREAD_FIBER_MANAGER_H

#include "fiber-manager.h"
#include <signal.h>
#include <list>
#include <unistd.h>
#include <setjmp.h>

namespace ns3 {

struct PthreadFiber;
class StackTrampoline;

enum PthreadFiberState
{
  RUNNING,
  SLEEP,
  DESTROY
};

class MemoryBounds
{
public:
  MemoryBounds ()
    : m_min (~0),
      m_max (0)
  {
  }
  void AddBound (void *address)
  {
    unsigned long v = (unsigned long) address;
    m_min = std::min (v, m_min);
    m_max = std::max (v, m_max);
  }
  void * GetStart (void) const
  {
    int size = getpagesize ();
    long start = m_min - (m_min % size);
    return (void*)start;
  }
  size_t GetSize (void) const
  {
    int size = getpagesize ();
    unsigned long start = m_min - (m_min % size);
    unsigned long end = ((m_max % size) == 0) ? m_max : (m_max + (size - (m_max % size)));
    return end - start;
  }
private:
  unsigned long m_min;
  unsigned long m_max;
};

struct PthreadFiberThread;

struct PthreadFiber : public Fiber
{
  struct PthreadFiberThread *thread;
  enum PthreadFiberState state;
  void *stack_copy;
  jmp_buf yield_env;
  size_t yield_stack_size;
  MemoryBounds stack_bounds;
};

struct PthreadFiberThread
{
  pthread_t thread;
  pthread_mutex_t mutex;
  pthread_cond_t condvar;
  uint32_t refcount;
  bool thread_started;
  jmp_buf initial_env;
  void (*func)(void *);
  void *context;
  size_t stack_size;
  StackTrampoline *trampoline;
  struct PthreadFiber *previous;
  struct PthreadFiber *next;
  MemoryBounds stack_bounds;
};


class PthreadFiberManager : public FiberManager
{
public:
  PthreadFiberManager ();
  virtual ~PthreadFiberManager ();
  virtual struct Fiber * Clone (struct Fiber *fiber);
  virtual struct Fiber *Create (void (*callback)(void *),
                                void *context,
                                uint32_t stackSize);
  virtual struct Fiber * CreateFromCaller (void);
  virtual void Delete (struct Fiber *fiber);
  virtual void SwitchTo (struct Fiber *from,
                         const struct Fiber *to);
  virtual uint32_t GetStackSize (struct Fiber *fiber) const;
  virtual void SetSwitchNotification (void (*fn)(void));
private:
  static void * Run (void *arg);
  void Yield (struct PthreadFiber *fiber);
  void Wakeup (struct PthreadFiber *fiber);
  void Start (struct PthreadFiber *fiber);
  static void * SelfStackBottom (void);
  void RestoreFiber (struct PthreadFiber *fiber);
  void (*m_notifySwitch)(void);
  StackTrampoline *m_trampoline;
};

} // namespace ns3

#endif /* PTHREAD_FIBER_MANAGER_H */
