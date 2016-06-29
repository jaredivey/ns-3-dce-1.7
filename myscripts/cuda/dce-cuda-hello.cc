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

NS_LOG_COMPONENT_DEFINE ("DceCudaHello");

int main (int argc, char *argv[])
{
  TIMER_TYPE t0, t1, t2;
  TIMER_NOW (t0);

  bool ocelot = false;
  uint32_t appChoice = 0;
  bool encrypt = true;
  std::string filename = "";
  std::string keyFilename = "key.txt";
  uint32_t numNodes = 1;
  std::ostringstream oss;

  CommandLine cmd;
  cmd.AddValue ("useOcelot", "Use Ocelot instead of CUDA", ocelot);
  cmd.AddValue ("appChoice", "Choose either: 0) matrixMul, 1) matrixMulDrv, 2) deviceQuery", appChoice);
  cmd.AddValue ("encrypt", "encrypt(1) or decrypt(0)", encrypt);
  cmd.AddValue ("filename", "file to encrypt or decrypt", filename);
  cmd.AddValue ("numNodes", "Number of nodes to simulate", numNodes);
  cmd.Parse (argc, argv);

  //LogComponentEnable ("ElfDependencies", LOG_LEVEL_DEBUG);
  NodeContainer nodes;
  nodes.Create (numNodes);

//  PointToPointHelper pointToPoint;
//  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
//  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));
//
//  NetDeviceContainer devices;
//  devices = pointToPoint.Install (nodes);

  InternetStackHelper stack;
  stack.Install (nodes);
//
//  Ipv4AddressHelper address;
//  address.SetBase ("10.1.1.0", "255.255.255.0");
//
//  Ipv4InterfaceContainer interfaces = address.Assign (devices);

  DceManagerHelper dceManager;
  dceManager.Install (nodes);

  ApplicationContainer apps;
  for (uint32_t i = 0; i < nodes.GetN(); ++i)
  {
	  DceApplicationHelper dce;

	  dce.SetStackSize (1<<20);
	  oss.str("");
	  switch (appChoice)
	  {
	  case 0:
		  oss << "matrixMul";
		  break;
	  case 1:
		  oss << "vectorAdd";
		  break;
	  case 2:
		  oss << "scan";
		  break;
	  case 3:
		  oss << "transpose";
		  break;
	  case 4:
		  oss << "mergeSort";
		  break;
	  case 5:
		  oss << "reduction";
		  break;
	  case 6:
		  oss << "deviceQuery";
		  break;
	  case 7:
		  oss << "aes";
		  break;
	  case 8:
		  oss << "matrixMulDrv";
		  break;
	  case 9:
		  oss << "vectorAddDrv";
		  break;
	  case 10:
		  oss << "simpleTextureDrv";
		  break;
	  case 11:
		  oss << "deviceQueryDrv";
		  break;
	  default:
		  break;
	  }
	  if (ocelot)
	  {
		  oss << "_ocelot";
	  }
	  else
	  {
		  oss << "_cuda";
	  }
	  dce.SetBinary (oss.str());
	  dce.ResetArguments ();
	  dce.ResetEnvironment ();
	  if (appChoice == 6)
	  {
	    if (encrypt) dce.AddArgument("ecb_encrypt");
	    else dce.AddArgument("ecb_decrypt");

	    dce.AddArgument(filename);
	    dce.AddArgument(keyFilename);
	  }
	  else if (ocelot)
	  {
	    dce.AddArgument("-device=1");
	  }

	  dce.AddEnvironment ("PATH", "/");

	  ApplicationContainer app;
	  app = dce.Install (nodes.Get(i));
	  app.Start (Seconds(i));
	  apps.Add(app);
  }

  TIMER_NOW (t1);
  Simulator::Run ();
  TIMER_NOW (t2);
  double d1 = TIMER_DIFF (t1, t0);
  double d2 = TIMER_DIFF (t2, t1);

  std::cout << oss.str() << "\t" << numNodes << "\t" << Simulator::Now().GetSeconds() << "\t" << d1 << "\t" << d2 << "\t" << d1+d2 << std::endl;
  Simulator::Destroy ();
}
