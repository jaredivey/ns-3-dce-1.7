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

enum LANGUAGE
{
  GCC,
  CLANG,
  PYTHON,
  JAVA
} LANGUAGE;

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

  int langChoice = GCC;
  int iteration = 0;
  int numNodes = 2;
  int numPings = 1000;
  int numBytes = 56;
  std::ostringstream oss;

  CommandLine cmd;
  cmd.AddValue ("langChoice", "Language/Compiler to execute", langChoice);
  cmd.AddValue ("numNodes", "Number of nodes to use", numNodes);
  cmd.AddValue ("numPings", "Number of times to ping a single address", numPings);
  cmd.AddValue ("numBytes", "Number of bytes to send each time", numBytes);
  cmd.AddValue ("iteration", "Dummy iteration variable", iteration);
  cmd.Parse (argc, argv);
std::cout << langChoice << "\t" << numPings << "\t" << numBytes << "\t" << numNodes << "\t";

  NodeContainer connNodes, ringNodes;
  connNodes.Create (numNodes);
  ringNodes.Create (numNodes);

  DceManagerHelper dceManager;

  InternetStackHelper stack;
  stack.Install (connNodes);
  stack.Install (ringNodes);

  PointToPointHelper connP2P;
  connP2P.SetDeviceAttribute ("DataRate", StringValue ("10Mbps"));
  connP2P.SetChannelAttribute ("Delay", StringValue ("1ms"));

  PointToPointHelper ringP2P;
  ringP2P.SetDeviceAttribute ("DataRate", StringValue ("100Mbps"));
  ringP2P.SetChannelAttribute ("Delay", StringValue ("1ms"));

  std::vector<NetDeviceContainer> connDevices;
  std::vector<NetDeviceContainer> ringDevices;
  for (int i = 0; i < numNodes; ++i)
  {
	  connDevices.push_back(connP2P.Install(connNodes.Get(i),ringNodes.Get(i)));
	  ringDevices.push_back(ringP2P.Install(ringNodes.Get(i),ringNodes.Get((i+1)%numNodes)));
  }

  Ipv4AddressHelper address;
  std::vector<Ipv4InterfaceContainer> connInterfaces;
  std::vector<Ipv4InterfaceContainer> ringInterfaces;

  oss.str ("");
  oss << "10.1.1.0";
  address.SetBase (oss.str ().c_str (), "255.255.255.0");
  connInterfaces.push_back(address.Assign (connDevices.at(0)));

  oss.str ("");
  oss << "10.1.2.0";
  address.SetBase (oss.str ().c_str (), "255.255.255.0");
  for (int i = 1; i < connDevices.size(); ++i)
  {
    connInterfaces.push_back(address.Assign (connDevices.at(i)));
    address.NewNetwork();
  }

  oss.str ("");
  oss << "192.168.1.0";
  address.SetBase (oss.str ().c_str (), "255.255.255.0");
  for (int i = 0; i < ringDevices.size(); ++i)
  {
    ringInterfaces.push_back(address.Assign (ringDevices.at(i)));
    address.NewNetwork();
  }
  // setup ip routes
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  dceManager.Install (connNodes);
  DceApplicationHelper dce;
  dce.SetStackSize (1<<20);

  ApplicationContainer pingApps;
  for (int i = 1; i < numNodes; ++i)
  {
    switch (langChoice)
    {
      case GCC:
        dce.SetBinary ("ping_gcc");
        dce.ResetArguments ();
        dce.ResetEnvironment ();
        break;
      case CLANG:
        dce.SetBinary ("ping_clang");
        dce.ResetArguments ();
        dce.ResetEnvironment ();
        break;
      case PYTHON:
        dce.SetBinary ("python2-dce");
        dce.ResetArguments ();
        dce.ResetEnvironment ();
        dce.AddEnvironment ("PATH", "/:/python2.7:/pox:/ryu");
        dce.AddEnvironment ("PYTHONHOME", "/:/python2.7:/pox:/ryu");
        dce.AddEnvironment ("PYTHONPATH", "/:/python2.7:/pox:/ryu");
        dce.AddArgument ("-S");
        dce.AddArgument ("testping.py");
        break;
      case JAVA:
        dce.SetBinary ("java");
        dce.ResetArguments ();
        dce.ResetEnvironment ();
        //dce.AddArgument ("-Xint");
        //dce.AddArgument ("-XX:ThreadStackSize=512");
        //dce.AddArgument ("-XX:ReservedCodeCacheSize=16m");
        //dce.AddArgument ("-XX:MaxInlineSize=0");
        //dce.AddArgument ("-XX:LoopUnrollLimit=0");
        //dce.AddArgument ("-XX:+UseCompressedOops");
        dce.AddArgument ("-cp");
        dce.AddArgument ("/");
        dce.AddArgument ("TestPing");
        break;
      default:
        std::cout << "Invalid language/compiler choice" << std::endl;
        return 1;
      }
    oss.str(""); connInterfaces.at(i).GetAddress(0).Print(oss);
    dce.AddArgument (oss.str());

    oss.str(""); oss << numBytes;
    dce.AddArgument (oss.str());

    oss.str(""); oss << numPings;
    dce.AddArgument (oss.str());

    ApplicationContainer pingApp = dce.Install (connNodes.Get (0));
    pingApp.Start (Seconds (i*numPings));
    pingApps.Add(pingApp);
  }

  //ApplicationContainer sinkApps;
  //PacketSinkHelper sink ("ns3::TcpSocketFactory",
  //                       InetSocketAddress (Ipv4Address::GetAny (), portStart));
  //sink.SetAttribute ("MaxBytes", UintegerValue(numBytes*numPings));
  //sinkApps.Add(sink.Install (connNodes.Get (0)));
  //sinkApps.Start (Seconds (0.0));

  TIMER_NOW (t1);
  Simulator::Run ();
  TIMER_NOW (t2);

  double d1 = TIMER_DIFF (t1, t0) + TIMER_DIFF (t2, t1);
  //uint32_t totalRx = 0;
  //for (uint32_t i = 0; i < sinkApps.GetN(); ++i)
  //  {
  //    Ptr<PacketSink> sink1 = DynamicCast<PacketSink> (sinkApps.Get (i));
  //    if (sink1)
  //      {
  //        totalRx += sink1->GetTotalRx();
  //      }
  //  }
  std::cout << Simulator::Now().GetSeconds() << " " << d1 << " " << ReportMemoryUsageMB () << " " << std::endl;
  Simulator::Destroy ();
}
