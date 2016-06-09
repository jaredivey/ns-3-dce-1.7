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
  int numNodes = 4;
  int loopCount = 1;
  int numBytes = 256;
  int portStart = 45000;
  std::ostringstream oss;

  CommandLine cmd;
  cmd.AddValue ("langChoice", "Language/Compiler to execute", langChoice);
  cmd.AddValue ("numNodes", "Number of nodes on each side", numNodes);
  cmd.AddValue ("loopCount", "Number of times to send data", loopCount);
  cmd.AddValue ("numBytes", "Number of bytes to send each time", numBytes);
  cmd.Parse (argc, argv);
std::cout << langChoice << "\t" << loopCount << "\t" << numBytes << "\t" << numNodes << "\t";

  NodeContainer leftNodes, rightNodes, linkNodes;
  rightNodes.Create (numNodes);
  leftNodes.Create (numNodes);
  linkNodes.Create (2);

  DceManagerHelper dceManager;

  InternetStackHelper stack;
  stack.Install (leftNodes);
  stack.Install (rightNodes);
  stack.Install (linkNodes);

  PointToPointHelper connP2P;
  connP2P.SetDeviceAttribute ("DataRate", StringValue ("10Mbps"));
  connP2P.SetChannelAttribute ("Delay", StringValue ("1ms"));

  PointToPointHelper linkP2P;
  linkP2P.SetDeviceAttribute ("DataRate", StringValue ("100Mbps"));
  linkP2P.SetChannelAttribute ("Delay", StringValue ("1ms"));

  std::vector<NetDeviceContainer> leftDevices;
  std::vector<NetDeviceContainer> rightDevices;
  std::vector<NetDeviceContainer> linkDevices;
  linkDevices.push_back(linkP2P.Install(linkNodes));
  for (int i = 0; i < numNodes; ++i)
  {
	  leftDevices.push_back(connP2P.Install(leftNodes.Get(i),linkNodes.Get(0)));
	  rightDevices.push_back(connP2P.Install(rightNodes.Get(i),linkNodes.Get(1)));
  }

  Ipv4AddressHelper address;
  std::vector<Ipv4InterfaceContainer> leftInterfaces;
  std::vector<Ipv4InterfaceContainer> rightInterfaces;
  std::vector<Ipv4InterfaceContainer> linkInterfaces;

  for (int i = 0; i < leftDevices.size(); ++i)
  {
	  oss.str ("");
	  oss << "10.1." << i+1 << ".0";
	  address.SetBase (oss.str ().c_str (), "255.255.255.0");

	  leftInterfaces.push_back(address.Assign (leftDevices.at(i)));
  }

  for (int i = 0; i < rightDevices.size(); ++i)
  {
	  oss.str ("");
	  oss << "10.2." << i+1 << ".0";
	  address.SetBase (oss.str ().c_str (), "255.255.255.0");

	  rightInterfaces.push_back(address.Assign (rightDevices.at(i)));
  }

  oss.str ("");
  oss << "192.168.1.0";
  address.SetBase (oss.str ().c_str (), "255.255.255.0");

  linkInterfaces.push_back(address.Assign (linkDevices.at(0)));

  // setup ip routes
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  dceManager.Install (leftNodes);
  DceApplicationHelper dce;
  dce.SetStackSize (1<<20);

  ApplicationContainer clientApps;
  for (int i = 0; i < leftNodes.GetN(); ++i)
  {
	  switch (langChoice)
	  {
	    case GCC:
	      dce.SetBinary ("client_gcc");
	      dce.ResetArguments ();
	      dce.ResetEnvironment ();
	      break;
	    case CLANG:
	      dce.SetBinary ("client_clang");
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
	      dce.AddArgument ("client.py");
	      break;
	    case JAVA:
	      dce.SetBinary ("java");
	      dce.ResetArguments ();
	      dce.ResetEnvironment ();
	      dce.AddArgument ("-Xint");
	      dce.AddArgument ("-XX:ThreadStackSize=512");
	      dce.AddArgument ("-XX:ReservedCodeCacheSize=16m");
	      dce.AddArgument ("-XX:MaxInlineSize=0");
	      dce.AddArgument ("-XX:LoopUnrollLimit=0");
	      dce.AddArgument ("-XX:+UseCompressedOops");
	      dce.AddArgument ("-cp");
	      dce.AddArgument ("/");
	      dce.AddArgument ("TCPClient");
	      break;
	    default:
	      std::cout << "Invalid language/compiler choice" << std::endl;
	      return 1;
	  }

      oss.str(""); rightInterfaces.at(i).GetAddress(0).Print(oss);
      dce.AddArgument (oss.str());

      oss.str(""); oss << portStart;
      dce.AddArgument (oss.str());

      oss.str(""); oss << numBytes;
      dce.AddArgument (oss.str());

      oss.str(""); oss << loopCount;
      dce.AddArgument (oss.str());
      ApplicationContainer client = dce.Install (leftNodes.Get (i));
      client.Start(Seconds (1.1));

      clientApps.Add(client);
  }

  TIMER_NOW (t1);
  Simulator::Run ();
  TIMER_NOW (t2);

  double d1 = TIMER_DIFF (t1, t0) + TIMER_DIFF (t2, t1);
  std::cout << Simulator::Now().GetSeconds() << " " << d1 << " " << ReportMemoryUsageMB () << " " << getpid () << std::endl;
  Simulator::Destroy ();
}
