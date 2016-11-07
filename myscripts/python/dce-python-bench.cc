#include "ns3/network-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/dce-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

using namespace ns3;

typedef struct timeval TIMER_TYPE;
#define TIMER_NOW(_t) gettimeofday (&_t,NULL);
#define TIMER_SECONDS(_t) ((double)(_t).tv_sec + (_t).tv_usec * 1e-6)
#define TIMER_DIFF(_t1, _t2) (TIMER_SECONDS (_t1) - TIMER_SECONDS (_t2))

enum BENCHMARK
{
  MATRIXMULT_GCC,
  MATRIXMULT_CLANG,
  MATRIXMULT_PYTHON,
  MATRIXMULT_JAVA,
  PIDIGITS_GCC,
  PIDIGITS_CLANG,
  PIDIGITS_PYTHON,
  PIDIGITS_JAVA,
  THREADTEST_GCC,
  THREADTEST_CLANG,
  THREADTEST_PYTHON,
  THREADTEST_JAVA,
  PING_GCC,
  PING_CLANG,
  PING_PYTHON,
  PING_JAVA,
  PYCUDA_DEMO,
  PYCUDA_DUMP
} BENCHMARK;

unsigned long
ReportMemoryUsage()
{
  pid_t pid;
  char work[4096];
  FILE* f;
  char* pCh;

  pid = getpid();
  sprintf(work, "/proc/%d/stat", (int)pid);
  f = fopen(work, "r");
  if (f == NULL)
    {
      std::cout <<"Can't open " << work << std::endl;
      return(0);
    }
  if(fgets(work, sizeof(work), f) == NULL)
    std::cout << "Error with fgets" << std::endl;
  fclose(f);
  strtok(work, " ");
  for (int i = 1; i < 23; i++)
    {
      pCh = strtok(NULL, " ");
    }
  return(atol(pCh));
}

unsigned long
ReportMemoryUsageMB()
{
  unsigned long u = ReportMemoryUsage();
  return ((u + 500000) / 1000000 );
}

int main (int argc, char *argv[])
{
  TIMER_TYPE t0, t1, t2;
  TIMER_NOW (t0);

  int benchmark = MATRIXMULT_GCC;
  int iteration = 0;
  int numBytes = 1000;
  int portStart = 45000;
  std::ostringstream oss;

  CommandLine cmd;
  cmd.AddValue ("benchmark", "Benchmark program to use", benchmark);
  cmd.AddValue ("iteration", "Dummy iteration variable", iteration);
  cmd.Parse (argc, argv);
  NodeContainer connNodes;
  connNodes.Create (1);

  DceManagerHelper dceManager;

  InternetStackHelper stack;
  stack.Install (connNodes);

  dceManager.Install (connNodes);

  DceApplicationHelper dce;
  dce.SetStackSize (1<<30);

  switch (benchmark)
  {
  case MATRIXMULT_GCC:
          std::cout << "matrixmult_gcc" << "\t";
	  dce.SetBinary ("matrixmult_gcc");
	  dce.ResetArguments ();
	  dce.ResetEnvironment ();
	  break;
  case PIDIGITS_GCC:
          std::cout << "pidigits_gcc" << "\t";
	  dce.SetBinary ("pidigits_gcc");
	  dce.ResetArguments ();
	  dce.ResetEnvironment ();
	  break;
  case THREADTEST_GCC:
          std::cout << "threadtest_gcc" << "\t";
	  dce.SetBinary ("threadtest_gcc");
	  dce.ResetArguments ();
	  dce.ResetEnvironment ();
	  break;
  case PING_GCC:
          std::cout << "ping_gcc" << "\t";
	  dce.SetBinary ("ping_gcc");
	  dce.ResetArguments ();
	  dce.ResetEnvironment ();
      dce.AddArgument ("127.0.0.1");
      dce.AddArgument ("56");
      dce.AddArgument ("1000");
	  break;
  case MATRIXMULT_CLANG:
          std::cout << "matrixmult_clang" << "\t";
	  dce.SetBinary ("matrixmult_clang");
	  dce.ResetArguments ();
	  dce.ResetEnvironment ();
	  break;
  case PIDIGITS_CLANG:
          std::cout << "pidigits_clang" << "\t";
	  dce.SetBinary ("pidigits_clang");
	  dce.ResetArguments ();
	  dce.ResetEnvironment ();
	  break;
  case THREADTEST_CLANG:
          std::cout << "threadtest_clang" << "\t";
	  dce.SetBinary ("threadtest_clang");
	  dce.ResetArguments ();
	  dce.ResetEnvironment ();
	  break;
  case PING_CLANG:
          std::cout << "ping_clang" << "\t";
	  dce.SetBinary ("ping_clang");
	  dce.ResetArguments ();
	  dce.ResetEnvironment ();
      dce.AddArgument ("127.0.0.1");
      dce.AddArgument ("56");
      dce.AddArgument ("1000");
	  break;
  case MATRIXMULT_PYTHON:
          std::cout << "matrixmult_python" << "\t";
	  dce.SetBinary ("python2-dce");
	  dce.ResetArguments ();
	  dce.ResetEnvironment ();
	  dce.AddEnvironment ("PATH", "/:/python2.7:/pox:/ryu");
	  dce.AddEnvironment ("PYTHONHOME", "/:/python2.7:/pox:/ryu");
	  dce.AddEnvironment ("PYTHONPATH", "/:/python2.7:/pox:/ryu");
	  dce.AddArgument ("-S");
	  dce.AddArgument ("matrixmult.py");
	  break;
  case PIDIGITS_PYTHON:
          std::cout << "pidigits_python" << "\t";
	  dce.SetBinary ("python2-dce");
	  dce.ResetArguments ();
	  dce.ResetEnvironment ();
	  dce.AddEnvironment ("PATH", "/:/python2.7:/pox:/ryu:/python2.7/lib-dynload");
	  dce.AddEnvironment ("PYTHONHOME", "/:/python2.7:/pox:/ryu");
	  dce.AddEnvironment ("PYTHONPATH", "/:/python2.7:/pox:/ryu:/python2.7/lib-dynload");
	  //dce.AddArgument ("-S");
	  //dce.AddArgument ("-u");
	  //dce.AddArgument ("-v");
	  //dce.AddArgument ("-d");
	  dce.AddArgument ("numpy_tutorial.py");
	  break;
  case THREADTEST_PYTHON:
          std::cout << "threadtest_python" << "\t";
	  dce.SetBinary ("python2-dce");
	  dce.ResetArguments ();
	  dce.ResetEnvironment ();
	  dce.AddEnvironment ("PATH", "/:/numpy:/python2.7:/pox:/ryu");
	  dce.AddEnvironment ("PYTHONHOME", "/:/numpy:/python2.7:/pox:/ryu");
	  dce.AddEnvironment ("PYTHONPATH", "/:/numpy:/python2.7:/pox:/ryu");
	  //dce.AddArgument ("-S");
	  dce.AddArgument ("threadtest.py");
	  break;
  case PING_PYTHON:
          std::cout << "ping_python" << "\t";
	  dce.SetBinary ("python2-dce");
	  dce.ResetArguments ();
	  dce.ResetEnvironment ();
	  dce.AddEnvironment ("PATH", "/:/python2.7:/pox:/ryu");
	  dce.AddEnvironment ("PYTHONHOME", "/:/python2.7:/pox:/ryu");
	  dce.AddEnvironment ("PYTHONPATH", "/:/python2.7:/pox:/ryu");
	  dce.AddArgument ("-S");
	  dce.AddArgument ("myping.py");
	  break;
  case MATRIXMULT_JAVA:
          std::cout << "matrixmult_java" << "\t";
	  dce.SetBinary ("java");
	  dce.ResetArguments ();
	  dce.ResetEnvironment ();
	  //dce.AddArgument ("-Xint");
	  dce.AddArgument ("-cp");
	  dce.AddArgument ("/");
	  dce.AddArgument ("MatrixMult");
	  break;
  case PIDIGITS_JAVA:
          std::cout << "pidigits_java" << "\t";
	  dce.SetBinary ("java");
	  dce.ResetArguments ();
	  dce.ResetEnvironment ();
	  //dce.AddArgument ("-Xint");
	  dce.AddArgument ("-cp");
	  dce.AddArgument ("/");
	  dce.AddArgument ("PiDigits");
	  break;
  case THREADTEST_JAVA:
          std::cout << "threadtest_java" << "\t";
	  dce.SetBinary ("java");
	  dce.ResetArguments ();
	  dce.ResetEnvironment ();
	  //dce.AddArgument ("-Xint");
	  dce.AddArgument ("-cp");
	  dce.AddArgument ("/");
	  dce.AddArgument ("ThreadTest");
	  break;
  case PING_JAVA:
          std::cout << "ping_java" << "\t";
	  dce.SetBinary ("java");
	  dce.ResetArguments ();
	  dce.ResetEnvironment ();
	  //dce.AddArgument ("-Xint");
	  dce.AddArgument ("-cp");
	  dce.AddArgument ("/");
	  dce.AddArgument ("Ping");
      dce.AddArgument ("127.0.0.1");
      dce.AddArgument ("56");
      dce.AddArgument ("1000");
	  break;
  case PYCUDA_DEMO:
          std::cout << "pycuda_demo" << "\t";
	  dce.SetBinary ("python2-dce");
	  dce.ResetArguments ();
	  dce.ResetEnvironment ();
	  dce.AddEnvironment ("PATH", "/:/python2.7:/pox:/ryu:/pycuda:/python2.7/lib-dynload");
	  dce.AddEnvironment ("PYTHONHOME", "/:/python2.7:/pox:/ryu:/pycuda");
	  dce.AddEnvironment ("PYTHONPATH", "/:/python2.7:/pox:/ryu:/pycuda:/python2.7/lib-dynload");
	  //dce.AddArgument ("-S");
	  //dce.AddArgument ("-u");
	  //dce.AddArgument ("-v");
	  //dce.AddArgument ("-d");
	  dce.AddArgument ("demo.py");
	  break;
  case PYCUDA_DUMP:
          std::cout << "pycuda_dump" << "\t";
	  dce.SetBinary ("python2-dce");
	  dce.ResetArguments ();
	  dce.ResetEnvironment ();
	  dce.AddEnvironment ("PATH", "/:/python2.7:/pox:/ryu:/pycuda:/python2.7/lib-dynload");
	  dce.AddEnvironment ("PYTHONHOME", "/:/python2.7:/pox:/ryu:/pycuda");
	  dce.AddEnvironment ("PYTHONPATH", "/:/python2.7:/pox:/ryu:/pycuda:/python2.7/lib-dynload");
	  //dce.AddArgument ("-S");
	  dce.AddArgument ("-u");
	  dce.AddArgument ("-v");
	  dce.AddArgument ("-d");
	  dce.AddArgument ("dump_properties.py");
	  break;
  default:
	  std::cout << "Invalid language/compiler choice" << std::endl;
	  return 1;
  }

  if (benchmark != PING_GCC && benchmark != PING_CLANG)
  {
    oss.str(""); oss << portStart;
    dce.AddArgument (oss.str());
  }

  ApplicationContainer client = dce.Install (connNodes.Get (0));
  client.Start(Seconds (1.1));

  PacketSinkHelper sink ("ns3::TcpSocketFactory",
                         InetSocketAddress (Ipv4Address::GetAny (), portStart));
  //sink.SetAttribute ("MaxBytes", UintegerValue (numBytes));
  ApplicationContainer sinkApps;
  sinkApps.Add(sink.Install (connNodes.Get (0)));
  sinkApps.Start (Seconds (0.0));

  TIMER_NOW (t1);
  Simulator::Run ();
  TIMER_NOW (t2);

  double d1 = TIMER_DIFF (t1, t0) + TIMER_DIFF (t2, t1);
  uint32_t totalRx = 0;
  for (uint32_t i = 0; i < sinkApps.GetN(); ++i)
    {
      Ptr<PacketSink> sink1 = DynamicCast<PacketSink> (sinkApps.Get (i));
      if (sink1)
        {
          totalRx += sink1->GetTotalRx();
        }
    }
  std::cout << Simulator::Now().GetSeconds() << " " << d1 << " " << totalRx << " " << ReportMemoryUsageMB () << " " << std::endl;
  Simulator::Destroy ();
}
