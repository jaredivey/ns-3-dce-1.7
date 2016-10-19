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

#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("DceCampusOFSwitch13");

enum CONN_TYPE {
	CSMA,
	P2P
} CONN_TYPE;

int
main (int argc, char *argv[])
{
  size_t nCampuses = 1;
  size_t nClientsPer = 42;
  size_t connType = P2P;
  bool verbose = false;
  bool trace = false;

  CommandLine cmd;
  cmd.AddValue ("campuses", "Number of campuses", nCampuses);
  cmd.AddValue ("connType", "Type of connection between controller and switch", connType);
  cmd.AddValue ("verbose", "Tell application to log if true", verbose);
  cmd.AddValue ("trace", "Tracing traffic to files", trace);
  cmd.Parse (argc, argv);

  if (verbose)
    {
      LogComponentEnable ("DceCampusOFSwitch13", LOG_LEVEL_ALL);
      LogComponentEnable ("OFSwitch13Helper", LOG_LEVEL_ALL);
      LogComponentEnable ("OFSwitch13Interface", LOG_LEVEL_ALL);
      LogComponentEnable ("OFSwitch13Device", LOG_LEVEL_ALL);
      LogComponentEnable ("OFSwitch13Port", LOG_LEVEL_ALL);
    }

  // Enabling Checksum computations
  GlobalValue::Bind ("ChecksumEnabled", BooleanValue (true));

  // Create controller first so its ID is 0 (i.e. will use files-0 space)
  NodeContainer of13ControllerNodes;
  of13ControllerNodes.Create(nCampuses);

  // Installing the tcp/ip stack into hosts
  InternetStackHelper internet;

  std::vector<NodeContainer> net1Servers;
  std::vector<Ipv4InterfaceContainer> net1Interfaces;
  std::vector<NodeContainer> net2Clients;
  std::vector<Ipv4InterfaceContainer> net22Interfaces;
  std::vector<Ipv4InterfaceContainer> net23Interfaces;
  std::vector<Ipv4InterfaceContainer> net24Interfaces;
  std::vector<Ipv4InterfaceContainer> net25Interfaces;
  std::vector<Ipv4InterfaceContainer> net26aInterfaces;
  std::vector<Ipv4InterfaceContainer> net26bInterfaces;
  std::vector<Ipv4InterfaceContainer> net26cInterfaces;
  std::vector<NodeContainer> net3Clients;
  std::vector<NodeContainer> net0Switches;
  std::vector<NodeContainer> net1Switches;
  std::vector<NodeContainer> net2Switches;
  std::vector<NodeContainer> net3Switches;
  std::vector<NodeContainer> net4Switch;
  std::vector<NodeContainer> net5Switch;
  for (uint32_t i = 0; i < nCampuses; ++i)
  {
	  // Create the host nodes
	  NodeContainer net1s;
	  net1s.Create(4);
	  net1Servers.push_back(net1s);
	  internet.Install(net1Servers[i]);
	  net1Interfaces.push_back(Ipv4InterfaceContainer());

	  NodeContainer net2c;
	  net2c.Create(7*nClientsPer);
	  net2Clients.push_back(net2c);
	  internet.Install(net2Clients[i]);
	  net22Interfaces.push_back(Ipv4InterfaceContainer());
	  net23Interfaces.push_back(Ipv4InterfaceContainer());
	  net24Interfaces.push_back(Ipv4InterfaceContainer());
	  net25Interfaces.push_back(Ipv4InterfaceContainer());
	  net26aInterfaces.push_back(Ipv4InterfaceContainer());
	  net26bInterfaces.push_back(Ipv4InterfaceContainer());
	  net26cInterfaces.push_back(Ipv4InterfaceContainer());

	  NodeContainer net3c;
	  net3c.Create(5*nClientsPer);
	  net3Clients.push_back(net3c);
	  internet.Install(net3Clients[i]);

	  // Create the switch nodes; stack is installed in ofswitch13 API
	  NodeContainer net0Sw;
	  net0Sw.Create(3);
	  net0Switches.push_back(net0Sw);

	  NodeContainer net1Sw;
	  net1Sw.Create(2);
	  net1Switches.push_back(net1Sw);

	  NodeContainer net2Sw;
	  net2Sw.Create(7);
	  net2Switches.push_back(net2Sw);

	  NodeContainer net3Sw;
	  net3Sw.Create(5);
	  net3Switches.push_back(net3Sw);

	  NodeContainer net4Sw;
	  net4Sw.Create(1);
	  net4Switch.push_back(net4Sw);

	  NodeContainer net5Sw;
	  net5Sw.Create(1);
	  net5Switch.push_back(net5Sw);
  }

  // Configure the CsmaHelper
  CsmaHelper csma_1gb5ms;
  csma_1gb5ms.SetChannelAttribute ("DataRate", DataRateValue (DataRate ("1Gbps")));
  csma_1gb5ms.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (5)));
  CsmaHelper csma_2gb200ms;
  csma_2gb200ms.SetChannelAttribute ("DataRate", DataRateValue (DataRate ("2Gbps")));
  csma_2gb200ms.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (200)));
  CsmaHelper csma_100mb1ms;
  csma_100mb1ms.SetChannelAttribute ("DataRate", DataRateValue (DataRate ("100Mbps")));
  csma_100mb1ms.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (1)));

  // Initialize IPv4 host helpers
  Ipv4AddressHelper ipv4switches;
  Ipv4InterfaceContainer internetIpIfaces;
  ipv4switches.SetBase ("192.168.0.0", "255.255.0.0");

  std::vector< Ptr<OFSwitch13Helper> > of13Helpers;
  for (uint32_t z = 0; z < nCampuses; ++z)
  {
	  // Set up controller
	  of13Helpers.push_back(CreateObject<OFSwitch13Helper> ());
	  if (connType == CSMA)
	  {
		  of13Helpers[z]->SetAttribute ("ChannelType", EnumValue (OFSwitch13Helper::SINGLECSMA));
	  }
	  else if (connType == P2P)
	  {
		  of13Helpers[z]->SetAttribute ("ChannelType", EnumValue (OFSwitch13Helper::DEDICATEDP2P));
	  }
	  of13Helpers[z]->InstallExternalController (of13ControllerNodes.Get(z));

      std::cout << "Creating Campus Network " << z << ":" << std::endl;
      // Create Net0
      std::cout << "  SubNet [ 0";
      NetDeviceContainer ndc0[3];
      NetDeviceContainer of13SwitchPorts0 [3];
      for (uint32_t i = 0; i < 3; ++i)
      {
          of13SwitchPorts0[i] = NetDeviceContainer ();
      }
      // Connect the three switches in a ring
      for (uint32_t i = 0; i < 3; ++i)
      {
    	  NodeContainer tmp0Sw;
    	  tmp0Sw.Add(net0Switches[z].Get(i));
    	  tmp0Sw.Add(net0Switches[z].Get((i+1)%3));
    	  ndc0[i] = csma_1gb5ms.Install(tmp0Sw);

          of13SwitchPorts0[i].Add(ndc0[i].Get(0));
          of13SwitchPorts0[(i+1)%3].Add(ndc0[i].Get(1));
      }

      // Create Net1
      std::cout << " 1";
      NetDeviceContainer ndc1[5];
      NetDeviceContainer of13SwitchPorts1 [2];
      for (uint32_t i = 0; i < 2; ++i)
      {
    	  of13SwitchPorts1[i] = NetDeviceContainer();
      }
      // Connect 2 servers to 1 switch
      for (uint32_t i = 0; i < 4; ++i)
      {
    	  NodeContainer tmp1;
    	  int swIndex = floor(i/2);
    	  tmp1.Add(net1Switches[z].Get(swIndex));
    	  tmp1.Add(net1Servers[z].Get(i));
    	  ndc1[i] = csma_1gb5ms.Install(tmp1);

    	  of13SwitchPorts1[swIndex].Add(ndc1[i].Get(0));
    	  net1Interfaces[z].Add(ipv4switches.Assign (NetDeviceContainer(ndc1[i].Get(1))));
      }
      // Connect the 2 switches together
      NodeContainer tmp11;
      tmp11.Add(net1Servers[z].Get(0));
      tmp11.Add(net1Servers[z].Get(1));
	  ndc1[4] = csma_1gb5ms.Install(tmp11);
	  of13SwitchPorts1[0].Add(ndc1[4].Get(0));
	  of13SwitchPorts1[1].Add(ndc1[4].Get(1));

	  // Connect Net0 to Net1
	  NodeContainer net0_1;
	  net0_1.Add(net0Switches[z].Get(2));
	  net0_1.Add(net1Switches[z].Get(0));
	  NetDeviceContainer ndc0_1;
	  ndc0_1 = csma_1gb5ms.Install(net0_1);
	  of13SwitchPorts0[2].Add(ndc0_1.Get(0));
	  of13SwitchPorts1[0].Add(ndc0_1.Get(1));

	  // Connect Net0 to Net4
      NetDeviceContainer of13SwitchPorts4 = NetDeviceContainer ();
	  NodeContainer net0_4;
	  net0_4.Add(net0Switches[z].Get(1));
	  net0_4.Add(net4Switch[z].Get(0));
	  NetDeviceContainer ndc0_4;
	  ndc0_4 = csma_1gb5ms.Install(net0_4);
	  of13SwitchPorts0[1].Add(ndc0_4.Get(0));
	  of13SwitchPorts4.Add(ndc0_4.Get(1));

	  // Connect Net0 to Net5
      NetDeviceContainer of13SwitchPorts5 = NetDeviceContainer ();
	  NodeContainer net0_5;
	  net0_5.Add(net0Switches[z].Get(1));
	  net0_5.Add(net5Switch[z].Get(0));
	  NetDeviceContainer ndc0_5;
	  ndc0_5 = csma_1gb5ms.Install(net0_5);
	  of13SwitchPorts0[1].Add(ndc0_5.Get(0));
	  of13SwitchPorts5.Add(ndc0_5.Get(1));

	  // Connect Net4 to Net5
	  NodeContainer net4_5;
	  net4_5.Add(net4Switch[z].Get(0));
	  net4_5.Add(net5Switch[z].Get(0));
	  NetDeviceContainer ndc4_5;
	  ndc4_5 = csma_1gb5ms.Install(net4_5);
	  of13SwitchPorts4.Add(ndc4_5.Get(0));
	  of13SwitchPorts5.Add(ndc4_5.Get(1));

	  // Create Net2
	  std::cout << " 2";
      NetDeviceContainer of13SwitchPorts2 [7];
      for (uint32_t i = 0; i < 7; ++i)
      {
    	  of13SwitchPorts2[i] = NetDeviceContainer();
      }

	  // Connect Net4 to Net2, switch 0
	  NodeContainer net4_20;
	  net4_20.Add(net4Switch[z].Get(0));
	  net4_20.Add(net2Switches[z].Get(0));
	  NetDeviceContainer ndc4_20;
	  ndc4_20 = csma_1gb5ms.Install(net4_20);
	  of13SwitchPorts4.Add(ndc4_20.Get(0));
	  of13SwitchPorts2[0].Add(ndc4_20.Get(1));

	  // Connect Net4 to Net2, switch 1
	  NodeContainer net4_21;
	  net4_21.Add(net4Switch[z].Get(0));
	  net4_21.Add(net2Switches[z].Get(1));
	  NetDeviceContainer ndc4_21;
	  ndc4_21 = csma_1gb5ms.Install(net4_21);
	  of13SwitchPorts4.Add(ndc4_21.Get(0));
	  of13SwitchPorts2[1].Add(ndc4_21.Get(1));

	  // Connect Net2 switches 0 and 1
	  NodeContainer net2_01;
	  net2_01.Add(net2Switches[z].Get(0));
	  net2_01.Add(net2Switches[z].Get(1));
	  NetDeviceContainer ndc2_01;
	  ndc2_01 = csma_1gb5ms.Install(net2_01);
	  of13SwitchPorts2[0].Add(ndc2_01.Get(0));
	  of13SwitchPorts2[1].Add(ndc2_01.Get(1));

	  // Connect Net2 switches 0 and 2
	  NodeContainer net2_02;
	  net2_02.Add(net2Switches[z].Get(0));
	  net2_02.Add(net2Switches[z].Get(2));
	  NetDeviceContainer ndc2_02;
	  ndc2_02 = csma_1gb5ms.Install(net2_02);
	  of13SwitchPorts2[0].Add(ndc2_02.Get(0));
	  of13SwitchPorts2[2].Add(ndc2_02.Get(1));

	  // Connect Net2 switches 1 and 3
	  NodeContainer net2_13;
	  net2_13.Add(net2Switches[z].Get(1));
	  net2_13.Add(net2Switches[z].Get(3));
	  NetDeviceContainer ndc2_13;
	  ndc2_13 = csma_1gb5ms.Install(net2_13);
	  of13SwitchPorts2[1].Add(ndc2_13.Get(0));
	  of13SwitchPorts2[3].Add(ndc2_13.Get(1));

	  // Connect Net2 switches 2 and 3
	  NodeContainer net2_23;
	  net2_23.Add(net2Switches[z].Get(2));
	  net2_23.Add(net2Switches[z].Get(3));
	  NetDeviceContainer ndc2_23;
	  ndc2_23 = csma_1gb5ms.Install(net2_23);
	  of13SwitchPorts2[2].Add(ndc2_23.Get(0));
	  of13SwitchPorts2[3].Add(ndc2_23.Get(1));

	  // Connect Net2 switches 2 and 4
	  NodeContainer net2_24;
	  net2_24.Add(net2Switches[z].Get(2));
	  net2_24.Add(net2Switches[z].Get(4));
	  NetDeviceContainer ndc2_24;
	  ndc2_24 = csma_1gb5ms.Install(net2_24);
	  of13SwitchPorts2[2].Add(ndc2_24.Get(0));
	  of13SwitchPorts2[4].Add(ndc2_24.Get(1));

	  // Connect Net2 switches 3 and 5
	  NodeContainer net2_35;
	  net2_35.Add(net2Switches[z].Get(3));
	  net2_35.Add(net2Switches[z].Get(5));
	  NetDeviceContainer ndc2_35;
	  ndc2_35 = csma_1gb5ms.Install(net2_35);
	  of13SwitchPorts2[3].Add(ndc2_35.Get(0));
	  of13SwitchPorts2[5].Add(ndc2_35.Get(1));

	  // Connect Net2 switches 5 and 6
	  NodeContainer net2_56;
	  net2_56.Add(net2Switches[z].Get(5));
	  net2_56.Add(net2Switches[z].Get(6));
	  NetDeviceContainer ndc2_56;
	  ndc2_56 = csma_1gb5ms.Install(net2_56);
	  of13SwitchPorts2[5].Add(ndc2_56.Get(0));
	  of13SwitchPorts2[6].Add(ndc2_56.Get(1));

	  // Connect Net2 switch 2 with 42 clients (slots 0-41)
	  // Connect Net2 switch 3 with 42 clients (slots 42-83)
	  // Connect Net2 switch 4 with 42 clients (slots 84-125)
	  // Connect Net2 switch 5 with 42 clients (slots 126-167)
	  for (uint32_t i = 0; i < nClientsPer; ++i)
	  {
		  NodeContainer net2_2c;
		  net2_2c.Add(net2Switches[z].Get(2));
		  net2_2c.Add(net2Clients[z].Get(i));
		  NetDeviceContainer ndc2_2c;
		  ndc2_2c = csma_100mb1ms.Install(net2_2c);
		  of13SwitchPorts2[2].Add(ndc2_2c.Get(0));
    	  net22Interfaces[z].Add(ipv4switches.Assign (NetDeviceContainer(ndc2_2c.Get(1))));

		  NodeContainer net2_3c;
		  net2_3c.Add(net2Switches[z].Get(3));
		  net2_3c.Add(net2Clients[z].Get(i+42));
		  NetDeviceContainer ndc2_3c;
		  ndc2_3c = csma_100mb1ms.Install(net2_3c);
		  of13SwitchPorts2[3].Add(ndc2_3c.Get(0));
    	  net23Interfaces[z].Add(ipv4switches.Assign (NetDeviceContainer(ndc2_3c.Get(1))));

		  NodeContainer net2_4c;
		  net2_4c.Add(net2Switches[z].Get(4));
		  net2_4c.Add(net2Clients[z].Get(i+42+42));
		  NetDeviceContainer ndc2_4c;
		  ndc2_4c = csma_100mb1ms.Install(net2_4c);
		  of13SwitchPorts2[4].Add(ndc2_4c.Get(0));
    	  net24Interfaces[z].Add(ipv4switches.Assign (NetDeviceContainer(ndc2_4c.Get(1))));

		  NodeContainer net2_5c;
		  net2_5c.Add(net2Switches[z].Get(5));
		  net2_5c.Add(net2Clients[z].Get(i+42+42+42));
		  NetDeviceContainer ndc2_5c;
		  ndc2_5c = csma_100mb1ms.Install(net2_5c);
		  of13SwitchPorts2[5].Add(ndc2_5c.Get(0));
    	  net25Interfaces[z].Add(ipv4switches.Assign (NetDeviceContainer(ndc2_5c.Get(1))));

		  NodeContainer net2_6ac;
		  net2_6ac.Add(net2Switches[z].Get(6));
		  net2_6ac.Add(net2Clients[z].Get(i+42+42+42+42));
		  NetDeviceContainer ndc2_6ac;
		  ndc2_6ac = csma_100mb1ms.Install(net2_6ac);
		  of13SwitchPorts2[6].Add(ndc2_6ac.Get(0));
    	  net26aInterfaces[z].Add(ipv4switches.Assign (NetDeviceContainer(ndc2_6ac.Get(1))));

		  NodeContainer net2_6bc;
		  net2_6bc.Add(net2Switches[z].Get(6));
		  net2_6bc.Add(net2Clients[z].Get(i+42+42+42+42+42));
		  NetDeviceContainer ndc2_6bc;
		  ndc2_6bc = csma_100mb1ms.Install(net2_6bc);
		  of13SwitchPorts2[6].Add(ndc2_6bc.Get(0));
    	  net26bInterfaces[z].Add(ipv4switches.Assign (NetDeviceContainer(ndc2_6bc.Get(1))));

		  NodeContainer net2_6cc;
		  net2_6cc.Add(net2Switches[z].Get(6));
		  net2_6cc.Add(net2Clients[z].Get(i+42+42+42+42+42+42));
		  NetDeviceContainer ndc2_6cc;
		  ndc2_6cc = csma_100mb1ms.Install(net2_6cc);
		  of13SwitchPorts2[6].Add(ndc2_6cc.Get(0));
    	  net26cInterfaces[z].Add(ipv4switches.Assign (NetDeviceContainer(ndc2_6cc.Get(1))));
	  }

	  // Create Net3
	  std::cout << " 3";
      NetDeviceContainer of13SwitchPorts3 [4];
      for (uint32_t i = 0; i < 4; ++i)
      {
    	  of13SwitchPorts3[i] = NetDeviceContainer();
      }


      // Install ports on all switches
      for (uint32_t i = 0; i < 3; ++i)
      {
    	  of13Helpers[z]->InstallSwitch (net0Switches[z].Get(i), of13SwitchPorts0 [i]);
      }

      // Enable datapath logs
      if (verbose)
        {
    	  of13Helpers[z]->EnableDatapathLogs ("all");
        }
      // Enable pcap traces
      if (trace)
        {
    	  of13Helpers[z]->EnableOpenFlowPcap ();
        }
  }

  DceManagerHelper dceManager;
  dceManager.Install (of13ControllerNodes);

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
  dce.AddArgument ("ryu/app/simple_switch_13_demo.py");
  //dce.AddArgument ("--ofp-tcp-listen-port");
  //dce.AddArgument ("6653");

  apps.Add (dce.Install (of13ControllerNodes.Get(0)));
  apps.Start (Seconds (0.0));

  // Send TCP traffic from host 0 to 1
  //Ipv4Address h1Addr = internetIpIfaces.GetAddress (1);
  //BulkSendHelper senderHelper ("ns3::TcpSocketFactory", InetSocketAddress (h1Addr, 8080));
  //senderHelper.SetAttribute ("MaxBytes", UintegerValue (0));
  //ApplicationContainer senderApp  = senderHelper.Install (hosts.Get (0));
  //senderApp.Start (Seconds (2));
  //PacketSinkHelper sinkHelper ("ns3::TcpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), 8080));
  //ApplicationContainer sinkApp = sinkHelper.Install (hosts.Get (1));
  //sinkApp.Start (Seconds (0));

  // Install FlowMonitor
  FlowMonitorHelper monitor;
  //monitor.Install (hosts);

  // Run the simulation for 30 seconds
  //Simulator::Stop (Seconds (10));
  //Simulator::Run ();
  //Simulator::Destroy ();

  // Transmitted bytes
  //Ptr<PacketSink> sink = DynamicCast<PacketSink> (sinkApp.Get (0));
  //std::cout << "Total bytes sent from H0 to H1: " << sink->GetTotalRx () << std::endl;

  // Dump FlowMonitor results
  monitor.SerializeToXmlFile ("FlowMonitor.xml", false, false);
}

