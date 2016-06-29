#include <vector>

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/dce-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/log.h"
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

using namespace ns3;

typedef struct timeval TIMER_TYPE;
#define TIMER_NOW(_t) gettimeofday (&_t,NULL);
#define TIMER_SECONDS(_t) ((double)(_t).tv_sec + (_t).tv_usec * 1e-6)
#define TIMER_DIFF(_t1, _t2) (TIMER_SECONDS (_t1) - TIMER_SECONDS (_t2))

NS_LOG_COMPONENT_DEFINE ("DceCudaAes");

int main (int argc, char *argv[])
{
  TIMER_TYPE t0, t1, t2;
  TIMER_NOW (t0);

  bool ocelot = false;
  std::string filename = "";
  std::string keyFilename = "key.txt";

  uint32_t numPairs = 1;
  std::ostringstream oss;

  CommandLine cmd;
  cmd.AddValue ("useOcelot", "Use Ocelot instead of CUDA", ocelot);
  cmd.AddValue ("numPairs", "Number of node pairs", numPairs);
  cmd.Parse (argc, argv);

  //LogComponentEnable ("ElfDependencies", LOG_LEVEL_DEBUG);

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("10Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("1ms"));

  std::vector<NodeContainer> nodePairs;
  std::vector<NetDeviceContainer> devicePairs;
  std::vector<Ipv4InterfaceContainer> interfacePairs;
  InternetStackHelper stack;
  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");

  DceManagerHelper dceManager;

  DceApplicationHelper dce;
  ApplicationContainer apps;

  for (uint32_t i = 0; i < numPairs; ++i)
  {
    NodeContainer nodes;
    nodes.Create (2);

    nodePairs.push_back(nodes);
    devicePairs.push_back(pointToPoint.Install (nodePairs.at(i)));

    stack.Install (nodePairs.at(i));
    interfacePairs.push_back(address.Assign (devicePairs.at(i)));

    dceManager.Install (nodePairs.at(i));

    dce.SetStackSize (1<<20);

    for (uint32_t j = 0; j < nodes.GetN(); ++j)
    {
      oss.str("");
      oss << "aes_dce";
//      if (ocelot)
//      {
//	    oss << "_ocelot";
//      }
//      else
//      {
//        oss << "_cuda";
//      }
      std::cout << "Binary to set: " << oss.str() << std::endl;
      dce.SetBinary (oss.str());
      dce.ResetArguments ();
      dce.ResetEnvironment ();
      if (j == 0)
      {
    	  dce.AddArgument("ecb_encrypt");
          dce.AddArgument("hello.txt");
      }
      else
      {
    	  dce.AddArgument("ecb_decrypt");
          dce.AddArgument("hello.txt.out");
      }
      dce.AddArgument(keyFilename);

      oss.str("");
      interfacePairs.at(i).GetAddress(1,0).Print(oss);
      dce.AddArgument(oss.str());
      if (ocelot)
      {
        dce.AddArgument("1");
      }
      else
      {
        dce.AddArgument("0");
      }

      dce.AddEnvironment ("PATH", "/");

      apps.Add(dce.Install (nodePairs.at(i).Get (j)));
    }
  }
  apps.Start (Seconds (0.0));

  Simulator::Stop (Seconds(30.0));
  TIMER_NOW (t1);
  Simulator::Run ();
  TIMER_NOW (t2);

  double d1 = TIMER_DIFF (t1, t0);
  double d2 = TIMER_DIFF (t2, t1);
  std::cout << oss.str() << "\t" << numPairs << "\t" << Simulator::Now().GetSeconds() << "\t" << d1 << "\t" << d2 << "\t" << d1+d2 << std::endl;
  Simulator::Destroy ();
}
