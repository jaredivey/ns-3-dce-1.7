/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 University of Campinas (Unicamp)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Luciano Chaves <luciano@lrc.ic.unicamp.br>
 *         Vitor M. Eichemberger <vitor.marge@gmail.com>
 *
 * Creating a chain of N  OpenFlow 1.3 switches and a single controller CTRL.
 * Traffic flows from host H0 to host H1.
 *
 *     H0                               H1
 *     |                                 |
 * ----------   ----------           ----------
 * |  Sw 0  |---|  Sw 1  |--- ... ---| Sw N-1 |
 * ----------   ----------           ----------
 *     :            :           :         :
 *     ...................... . . . .......
 *                       :
 *                      CTRL
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/ofswitch13-module.h"
#include "ns3/network-module.h"
#include "ns3/dce-module.h"
#include "ns3/tap-bridge-module.h"

#include <vector>

using namespace ns3;

typedef struct timeval TIMER_TYPE;
#define TIMER_NOW(_t) gettimeofday (&_t,NULL);
#define TIMER_SECONDS(_t) ((double)(_t).tv_sec + (_t).tv_usec * 1e-6)
#define TIMER_DIFF(_t1, _t2) (TIMER_SECONDS (_t1) - TIMER_SECONDS (_t2))

NS_LOG_COMPONENT_DEFINE ("DceChainOFSwitch13");

enum CONN_TYPE {
	CSMA,
	P2P
} CONN_TYPE;

enum APP_TYPE {
	SS,
	FWM,
	NMBFS,
	NMUCS,
	FWCM,
	FWS,
	NSBFS,
	NSUCS,
	FWCS
};

static const char *apps[] =
{
		"ss",
		"fwm",
		"nm-bfs",
		"nm-ucs",
		"fwcm",
		"fws",
		"ns-bfs",
		"ns-ucs",
		"fwcs"
};

int
main (int argc, char *argv[])
{
  TIMER_TYPE t0, t1, t2;
  TIMER_NOW (t0);

  size_t nSwitches = 8;
  size_t nHosts = 8;
  size_t connType = CSMA;
  bool verbose = false;
  bool trace = false;
  uint32_t app = 0;
  bool realController = false;
  size_t numFlows = 0;
  uint32_t run = 0;

  CommandLine cmd;
  cmd.AddValue ("switches", "Number of OpenFlow switches", nSwitches);
  cmd.AddValue ("connType", "Type of connection between controller and switch", connType);
  cmd.AddValue ("verbose", "Tell application to log if true", verbose);
  cmd.AddValue ("trace", "Tracing traffic to files", trace);
  cmd.AddValue ("app", "Which application to use", app);
  cmd.AddValue ("realController", "Use external controller through TAP", realController);
  cmd.AddValue ("numFlows", "Number of flows (x4) to transmit", numFlows);
  cmd.AddValue ("run", "Adjust the run value", run);
  cmd.Parse (argc, argv);

  std::cout << apps[app] << "\t" << realController << "\t" << numFlows << "\t";
  RngSeedManager::SetRun(run);

  //LogComponentEnable ("Dce", LOG_LEVEL_INFO);
  //LogComponentEnable ("DceTime", LOG_LEVEL_INFO);
  if (verbose)
    {
      LogComponentEnable ("ChainOFSwitch13", LOG_LEVEL_ALL);
      LogComponentEnable ("OFSwitch13Helper", LOG_LEVEL_ALL);
      LogComponentEnable ("OFSwitch13Controller", LOG_LEVEL_ALL);
      LogComponentEnable ("OFSwitch13LearningController", LOG_LEVEL_ALL);
      LogComponentEnable ("OFSwitch13Interface", LOG_LEVEL_ALL);
      LogComponentEnable ("OFSwitch13Device", LOG_LEVEL_ALL);
      LogComponentEnable ("OFSwitch13Port", LOG_LEVEL_ALL);
    }

  // Enabling Checksum computations
  if (realController)
  {
	  GlobalValue::Bind ("SimulatorImplementationType", StringValue ("ns3::RealtimeSimulatorImpl"));
	  GlobalValue::Bind ("ChecksumEnabled", BooleanValue (true));
  }
  else
  {
	  GlobalValue::Bind ("ChecksumEnabled", BooleanValue (false));
  }

  // Create controller first so its ID is 0 (i.e. will use files-0 space)
  Ptr<Node> of13ControllerNode = CreateObject<Node> ();

  // Create the host nodes
  std::vector<NodeContainer> hosts;
  for (size_t i = 0; i < nSwitches; ++i)
  {
	  NodeContainer hostSet;
	  hostSet.Create (nHosts);
	  hosts.push_back(hostSet);
  }

  // Create the switches nodes
  NodeContainer of13SwitchNodes;
  of13SwitchNodes.Create (nSwitches);

  // Configure the CsmaHelper
  CsmaHelper csmaHelper;
  csmaHelper.SetChannelAttribute ("DataRate", DataRateValue (DataRate ("1Gbps")));
  csmaHelper.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (1)));

  NetDeviceContainer hostDevices [nSwitches];
  NetDeviceContainer of13SwitchPorts [nSwitches];
  for (size_t i = 1; i < nSwitches; i++)
    {
      of13SwitchPorts [i] = NetDeviceContainer ();
      hostDevices [i] = NetDeviceContainer();
    }

  for (size_t i = 0; i < nSwitches; i++)
    {
	  // Connect hosts to each switch
	  for (size_t j = 0; j < nHosts; ++j)
	  {
		  NodeContainer ncH0 (hosts.at(i).Get (j), of13SwitchNodes.Get (i));
		  NetDeviceContainer linkH0 = csmaHelper.Install (ncH0);
		  hostDevices[i].Add (linkH0.Get (0));
		  of13SwitchPorts[i].Add (linkH0.Get (1));
	  }

	  // Connect the switches in chain
	  if (i > 0)
	  {
	      NodeContainer nc (of13SwitchNodes.Get (i - 1), of13SwitchNodes.Get (i));
	      NetDeviceContainer link = csmaHelper.Install (nc);
	      of13SwitchPorts [i - 1].Add (link.Get (0));
	      of13SwitchPorts [i].Add (link.Get (1));
	  }
    }

  // Installing the tcp/ip stack into hosts
  InternetStackHelper internet;
  for (size_t i = 0; i < nSwitches; ++i)
  {
	  internet.Install (hosts.at(i));
  }

  Ptr<OFSwitch13Helper> of13Helper = CreateObject<OFSwitch13Helper> ();
  if (realController)
  {
	  of13Helper->SetAttribute ("ChannelType", EnumValue (OFSwitch13Helper::DEDICATEDCSMA));
  }
  else
  {
	  of13Helper->SetAttribute ("ChannelType", EnumValue (OFSwitch13Helper::DEDICATEDP2P));
  }
  of13Helper->InstallExternalController (of13ControllerNode);

  // Install OpenFlow device in every switch
  OFSwitch13DeviceContainer of13SwitchDevices;
  for (size_t i = 0; i < nSwitches; i++)
    {
      of13SwitchDevices = of13Helper->InstallSwitch (of13SwitchNodes.Get (i), of13SwitchPorts [i]);
    }

  DceManagerHelper dceManager;
  dceManager.Install (of13ControllerNode, 100);
  //dceManager.SetDelayModel("ns3::RandomProcessDelayModel", "Variable", StringValue ("ns3::GammaRandomVariable[Alpha=1,Beta=2]"));

  std::stringstream ss;

  // Set IPv4 host address
  Ipv4AddressHelper ipv4switches;
  std::vector<Ipv4InterfaceContainer> internetIpIfaces;
  ss.str("");
  ss << "10.1.0.0";
  ipv4switches.SetBase (ss.str().c_str(), "255.255.0.0");
  for (uint32_t i = 0; i < nSwitches; ++i)
  {
	  internetIpIfaces.push_back(ipv4switches.Assign (hostDevices[i]));
  }

  // Set up controller node application
  DceApplicationHelper dce;
  ApplicationContainer apps;

  // Set up controller node application
  if (realController)
  {
	  // TapBridge to local machine
	  // The default configuration expects a controller on you local machine at port 6653
	  TapBridgeHelper tapBridge;
	  tapBridge.SetAttribute ("Mode", StringValue ("ConfigureLocal"));
	  for (uint32_t tapIdx = 0; tapIdx < of13Helper->m_ctrlDevs.GetN(); ++tapIdx)
	  {
		  ss.str("");
		  ss << "ctrl" << tapIdx;
    	  tapBridge.Install (of13ControllerNode,
    			  of13Helper->m_ctrlDevs.Get(tapIdx),
    			  StringValue (ss.str()));
	  }
  }
  else
  {
      DceApplicationHelper dce;

      dce.SetStackSize (1<<30);
      dce.SetBinary ("python2-dce");
      dce.ResetArguments ();
      dce.ResetEnvironment ();
      dce.AddEnvironment ("PATH", "/:/python2.7:/pox:/ryu");
      dce.AddEnvironment ("PYTHONHOME", "/:/python2.7:/pox:/ryu");
      dce.AddEnvironment ("PYTHONPATH", "/:/python2.7:/pox:/ryu");
      if (verbose)
      {
    	  dce.AddArgument ("-v");
      }
      dce.AddArgument ("ryu-manager");
      if (verbose)
      {
    	  dce.AddArgument ("--verbose");
      }
      switch (app)
      {
      case SS:
    	  dce.AddArgument("ryu/app/simple_switch_13.py");
    	  break;
      case FWS:
    	  dce.AddArgument("ryu/app/fw_simple.py");
    	  break;
      case FWM:
    	  dce.AddArgument("ryu/app/fw_mpls.py");
    	  break;
      case NSBFS:
    	  dce.AddArgument("ryu/app/nix_simple_bfs.py");
    	  break;
      case NSUCS:
    	  dce.AddArgument("ryu/app/nix_simple_ucs.py");
    	  break;
      case NMBFS:
    	  dce.AddArgument("ryu/app/nix_mpls_bfs.py");
    	  break;
      case NMUCS:
    	  dce.AddArgument("ryu/app/nix_mpls_ucs.py");
    	  break;
      case FWCS:
    	  dce.AddArgument("ryu/app/fw_cuda_simple.py");
    	  break;
      case FWCM:
    	  dce.AddArgument("ryu/app/fw_cuda_mpls.py");
    	  break;
      default:
    	  NS_LOG_ERROR ("Invalid controller application");
      }

      apps.Add (dce.Install (of13ControllerNode));
  }
  apps.Start (Seconds (0.0));

  Config::SetDefault ("ns3::OnOffApplication::OnTime",
                      StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
  Config::SetDefault ("ns3::OnOffApplication::OffTime",
                      StringValue ("ns3::ConstantRandomVariable[Constant=0]"));
  Config::SetDefault ("ns3::OnOffApplication::DataRate",
		  	  	  	  DataRateValue(DataRate("100kb/s")));
  Config::SetDefault ("ns3::OnOffApplication::PacketSize",
		  	  	  	  UintegerValue(1400));

  ApplicationContainer clientApps;
  ApplicationContainer sinkApps;
  Ptr<NormalRandomVariable> nrng = CreateObject<NormalRandomVariable> ();
  double startTime1 = 0.0;
  double startTime2 = 0.0;
  double startTime3 = 0.0;
  if (numFlows > 0)
  {
	  for (size_t i = 0; i < 4; ++i)
	  {
		  // First flow 10.1.1.* to 10.1.6.*
		  OnOffHelper client ("ns3::UdpSocketFactory", Address ());
		  AddressValue remoteAddress (InetSocketAddress (internetIpIfaces.at(6).GetAddress(i,0), 45000));
		  client.SetAttribute ("Remote", remoteAddress);

		  ApplicationContainer clientApp;
		  clientApp.Add (client.Install (hosts.at(1).Get(i)));
		  startTime1 = nrng->GetValue(7.8388,1.0597);
		  std::cout << startTime1 << "\t";
		  clientApp.Start (Seconds (startTime1));
		  clientApps.Add(clientApp);

		  PacketSinkHelper sinkHelper ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), 45000));
		  sinkApps.Add(sinkHelper.Install (hosts.at(6).Get (i)));
	  }
	  startTime3 = startTime1; // In case fewer flows
  }
  if (numFlows > 1)
  {
	  for (size_t i = 0; i < 4; ++i)
	  {
		  // First flow 10.1.1.* to 10.1.6.*
		  OnOffHelper client ("ns3::UdpSocketFactory", Address ());
		  AddressValue remoteAddress (InetSocketAddress (internetIpIfaces.at(5).GetAddress(i,0), 45000));
		  client.SetAttribute ("Remote", remoteAddress);

		  ApplicationContainer clientApp;
		  clientApp.Add (client.Install (hosts.at(2).Get(i)));
		  startTime2 = nrng->GetValue(13.1000,1.2928);
		  std::cout << startTime2 << "\t";
		  clientApp.Start (Seconds (startTime2));
		  clientApps.Add(clientApp);

		  PacketSinkHelper sinkHelper ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), 45000));
		  sinkApps.Add(sinkHelper.Install (hosts.at(5).Get (i)));
	  }
	  startTime3 = startTime2; // In case fewer flows
  }
  if (numFlows > 2)
  {
	  for (size_t i = 0; i < 4; ++i)
	  {
		  // First flow 10.1.1.* to 10.1.6.*
		  OnOffHelper client ("ns3::UdpSocketFactory", Address ());
		  AddressValue remoteAddress (InetSocketAddress (internetIpIfaces.at(4).GetAddress(i,0), 45000));
		  client.SetAttribute ("Remote", remoteAddress);

		  ApplicationContainer clientApp;
		  clientApp.Add (client.Install (hosts.at(3).Get(i)));
		  startTime3 = nrng->GetValue(18.7396, 1.6237);
		  std::cout << startTime3 << "\t";
		  clientApp.Start (Seconds (startTime3));
		  clientApps.Add(clientApp);

		  PacketSinkHelper sinkHelper ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), 45000));
		  sinkApps.Add(sinkHelper.Install (hosts.at(4).Get (i)));
	  }
  }
  if (numFlows > 0)
  {
	  sinkApps.Start (Seconds (0));
  }

  // Send ping from host 0 to 1
  // "Arping" to allow Ryu to know the source and dest that we really want to send to
//  V4PingHelper arping1(internetIpIfaces.at(nSwitches-1).GetAddress(1,0));
//  arping1.SetAttribute("Verbose", BooleanValue(true));
//  arping1.SetAttribute("Count", UintegerValue(1));
//  ApplicationContainer arpingApp1 = arping1.Install(hosts.at(0).Get(0));
//  arpingApp1.Start(Seconds(2));
//
//  V4PingHelper arping2(internetIpIfaces.at(0).GetAddress(1,0));
//  arping2.SetAttribute("Verbose", BooleanValue(true));
//  arping2.SetAttribute("Count", UintegerValue(1));
//  ApplicationContainer arpingApp2 = arping2.Install(hosts.at(nSwitches-1).Get(0));
//  arpingApp2.Start(Seconds(3));

  V4PingHelper v4ping(internetIpIfaces.at(nSwitches-1).GetAddress(0,0));
  v4ping.SetAttribute("Verbose", BooleanValue(true));
  v4ping.SetAttribute("Stopper", BooleanValue(true));
  v4ping.SetAttribute("Size", UintegerValue(1422));
  v4ping.SetAttribute("Count", UintegerValue(2));
  ApplicationContainer pingApps = v4ping.Install(hosts.at(0).Get(0));
  pingApps.Start(Seconds(startTime3+20.0));
  std::cout << startTime3+20.0 << std::endl;

  // Enable datapath logs
  if (verbose)
    {
      of13Helper->EnableDatapathLogs ("all");
    }

  // Run the simulation for 30 seconds
  Simulator::Stop(Seconds(startTime3+50.0));
  Simulator::Run ();
  if (!realController)
  {
	  Ptr<DceApplication> controller = DynamicCast<DceApplication> (of13ControllerNode->GetApplication(0));
	  controller->StopExternally();
  }

  // Transmitted bytes
  uint32_t sentBytes = 0;
  uint32_t recvBytes = 0;
  for (size_t i = 0; i < clientApps.GetN(); ++i)
  {
	  Ptr<OnOffApplication> source = DynamicCast<OnOffApplication> (clientApps.Get (i));
	  sentBytes += source->m_totBytes;
  }
  for (size_t i = 0; i < sinkApps.GetN(); ++i)
  {
	  Ptr<PacketSink> sink = DynamicCast<PacketSink> (sinkApps.Get (i));
	  recvBytes += sink->GetTotalRx();
  }
  TIMER_NOW(t1);
  double d1 = TIMER_DIFF (t1, t0);
  std::cout << recvBytes << "\t" << (sentBytes == 0 ? 0 : 100 * (sentBytes - recvBytes) / sentBytes) << "%\t"
		  << "\t" << recvBytes * 8.0 / Simulator::Now().GetSeconds() << "\t" << Simulator::Now().GetSeconds()
		  << "\t" << d1 << std::endl;

  Simulator::Destroy ();
}

