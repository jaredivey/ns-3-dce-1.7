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

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("DceChainOFSwitch13");

enum CONN_TYPE {
	CSMA,
	P2P
} CONN_TYPE;

int
main (int argc, char *argv[])
{
  size_t nSwitches = 1;
  size_t connType = CSMA;
  bool stp = false;
  bool verbose = false;
  bool trace = false;
  bool nix = true;

  CommandLine cmd;
  cmd.AddValue ("switches", "Number of OpenFlow switches", nSwitches);
  cmd.AddValue ("connType", "Type of connection between controller and switch", connType);
  cmd.AddValue ("stp", "Use spanning tree with controller", stp);
  cmd.AddValue ("verbose", "Tell application to log if true", verbose);
  cmd.AddValue ("trace", "Tracing traffic to files", trace);
  cmd.AddValue ("nix", "Use NIx controller app", nix);
  cmd.Parse (argc, argv);

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
  GlobalValue::Bind ("ChecksumEnabled", BooleanValue (true));

  // Create controller first so its ID is 0 (i.e. will use files-0 space)
  Ptr<Node> of13ControllerNode = CreateObject<Node> ();

  // Create the host nodes
  NodeContainer hosts;
  hosts.Create (2);

  // Create the switches nodes
  NodeContainer of13SwitchNodes;
  of13SwitchNodes.Create (nSwitches);

  // Configure the CsmaHelper
  CsmaHelper csmaHelper;
  csmaHelper.SetChannelAttribute ("DataRate", DataRateValue (DataRate ("100Mbps")));
  csmaHelper.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));

  NetDeviceContainer hostDevices;
  NetDeviceContainer of13SwitchPorts [nSwitches];
  for (size_t i = 1; i < nSwitches; i++)
    {
      of13SwitchPorts [i] = NetDeviceContainer ();
    }

  // Connect H0 to first switch
  NodeContainer ncH0 (hosts.Get (0), of13SwitchNodes.Get (0));
  NetDeviceContainer linkH0 = csmaHelper.Install (ncH0);
  hostDevices.Add (linkH0.Get (0));
  of13SwitchPorts [0].Add (linkH0.Get (1));

  // Connect H1 to last switch
  NodeContainer ncH1 (hosts.Get (1), of13SwitchNodes.Get (nSwitches - 1));
  NetDeviceContainer linkH1 = csmaHelper.Install (ncH1);
  hostDevices.Add (linkH1.Get (0));
  of13SwitchPorts [nSwitches - 1].Add (linkH1.Get (1));

  // Connect the switches in chain
  for (size_t i = 1; i < nSwitches; i++)
    {
      NodeContainer nc (of13SwitchNodes.Get (i - 1), of13SwitchNodes.Get (i));
      NetDeviceContainer link = csmaHelper.Install (nc);
      of13SwitchPorts [i - 1].Add (link.Get (0));
      of13SwitchPorts [i].Add (link.Get (1));

      // Connect the last switch to the first in a loop if testing spanning tree
      if (stp && (i == nSwitches - 1))
      {
          NodeContainer nc (of13SwitchNodes.Get (i), of13SwitchNodes.Get (0));
          NetDeviceContainer link = csmaHelper.Install (nc);
          of13SwitchPorts [i].Add (link.Get (0));
          of13SwitchPorts [0].Add (link.Get (1));
      }
    }

  // Installing the tcp/ip stack into hosts
  InternetStackHelper internet;
  internet.Install (hosts);

  Ptr<OFSwitch13Helper> of13Helper = CreateObject<OFSwitch13Helper> ();
  if (connType == CSMA)
  {
	  of13Helper->SetAttribute ("ChannelType", EnumValue (OFSwitch13Helper::SINGLECSMA));
  }
  else if (connType == P2P)
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
  dceManager.Install (of13ControllerNode);

  // Set IPv4 host address
  Ipv4AddressHelper ipv4switches;
  Ipv4InterfaceContainer internetIpIfaces;
  ipv4switches.SetBase ("10.1.1.0", "255.255.255.0");
  internetIpIfaces = ipv4switches.Assign (hostDevices);

  // Set up controller node application
  DceApplicationHelper dce;
  ApplicationContainer apps;

  dce.SetStackSize (1<<20);
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
  if (nix)
  {
	  dce.AddArgument ("ryu/app/simple_switch_13_demo.py");
  }
  else
  {
	  dce.AddArgument ("ryu/app/simple_switch_13_stp.py");
  }
  //dce.AddArgument ("--ofp-tcp-listen-port");
  //dce.AddArgument ("6653");

  apps.Add (dce.Install (of13ControllerNode));
  apps.Start (Seconds (0.0));


  // Send TCP traffic from host 0 to 1
  Ipv4Address h1Addr = internetIpIfaces.GetAddress (1);
  BulkSendHelper senderHelper ("ns3::TcpSocketFactory", InetSocketAddress (h1Addr, 8080));
  senderHelper.SetAttribute ("MaxBytes", UintegerValue (0));
  ApplicationContainer senderApp  = senderHelper.Install (hosts.Get (0));
  senderApp.Start (Seconds (2));
  PacketSinkHelper sinkHelper ("ns3::TcpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), 8080));
  ApplicationContainer sinkApp = sinkHelper.Install (hosts.Get (1));
  sinkApp.Start (Seconds (0));

  // Enable datapath logs
  if (verbose)
    {
      of13Helper->EnableDatapathLogs ("all");
    }

  // Enable pcap traces
  if (trace)
    {
      of13Helper->EnableOpenFlowPcap ();
      csmaHelper.EnablePcap ("ofswitch", of13SwitchNodes, true);
      csmaHelper.EnablePcap ("host", hostDevices);
    }

  // Install FlowMonitor
  FlowMonitorHelper monitor;
  monitor.Install (hosts);

  // Run the simulation for 30 seconds
  Simulator::Stop (Seconds (10));
  Simulator::Run ();
  Simulator::Destroy ();

  // Transmitted bytes
  Ptr<PacketSink> sink = DynamicCast<PacketSink> (sinkApp.Get (0));
  std::cout << "Total bytes sent from H0 to H1: " << sink->GetTotalRx () << std::endl;

  // Dump FlowMonitor results
  monitor.SerializeToXmlFile ("FlowMonitor.xml", false, false);
}

