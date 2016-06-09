import subprocess
import os
import signal
import shlex
import time

import sys
from subprocess import PIPE, Popen
        
#file3 is the file store the final result    
file3 = "simulationResult/wns3-ping-resource-final.txt"
#clear the file
out3 = open(file3,'wb')
out3.close();
out3 = open(file3,'ab')

file4 = "simulationResult/wns3-ping-result-final.txt"
out4 = open(file4,'wb')
out4.close();
out4 = open(file4,'ab')

def ToFloat(line,index):
    line_split = line.split()
    if line_split[index] == "KiB":
        temp = float(line_split[index-1])
    elif line_split[index] == "MiB":
        temp = float(line_split[index-1]) * 1024
    elif line_split[index] == "GiB":
        temp = float(line_split[index-1]) * 1024 * 1024
    else:
        print "wrong format\n"
    return temp

for langChoice in range(3,4):
    for numNodes in range(10,51,10):
        for iteration in range(0,10):
            out3.write("processing language = {0}, numNodes = {1}, iteration = {2}, at {3}\n".format(langChoice,numNodes,iteration,time.strftime('%Y-%m-%d %X',time.localtime(time.time()))))
            out3.flush()

            print "processing language = {0}, numNodes = {1}, iteration = {2}, at {3}".format(langChoice,numNodes,iteration,time.strftime('%Y-%m-%d %X',time.localtime(time.time())))
            cmd1 = "./waf --run 'dce-python-ping --langChoice=" + str(langChoice) +" --numNodes=" + str(numNodes) + " --iteration=" + str(iteration) + "'"
            print cmd1
            p1 = subprocess.Popen(shlex.split(cmd1),stdout=out4)
            while True:
                try:
                    pid_pox = subprocess.check_output(["pgrep", "-f", "/dce-python-ping"])
                    #pid_pox=subprocess.check_output(["./GetPid", str(langChoice), str(numNodes), str(iteration)])
                    break
                except subprocess.CalledProcessError:
                    time.sleep(1)
                    pass
            print pid_pox
            #out4.write(str(langChoice)+ " " + str(numNodes) + " " + str(pid_pox))
            #out4.flush()

            
            max_private = 0
            max_share = 0
            max_total = 0
            is_break = False
            while True:
                p_mem = subprocess.Popen(["python","-u","ps_mem.py",'-p', pid_pox],bufsize=1, stdin=open(os.devnull), stdout=PIPE)
                temp_private = 0
                temp_share = 0
                temp_total = 0 
                for line in iter(p_mem.stdout.readline, ''):
                    out3.write(line)          # print to stdout immediately
                    out3.flush()
                    if line.find("python") != -1:                   
                        temp_private = ToFloat(line,1)
                        temp_share = ToFloat(line,4)
                        temp_total = ToFloat(line,7)
                        if temp_total > max_total:
                            max_total = temp_total
                            max_private = temp_private
                            max_share = temp_share           
                print "temp_total = {0}, max_total = {1} \n".format(temp_total, max_total)
                try:
                    pid_pox = subprocess.check_output(["pgrep", "-f", "/dce-python-ping"])
                    #pid_pox=subprocess.check_output(["./GetPid", str(langChoice), str(numNodes), str(iteration)])
                except subprocess.CalledProcessError:
                    break
                p_mem.stdout.close()
                p_mem.wait()
                time.sleep(2)
            
            out4.write(str(max_private)+' '+str(max_share)+' '+str(max_total)+'\n')
            out4.flush()

            p1.wait() 
            p_mem.wait()
        out4.write("\n")
        out4.flush()
out3.close()
out4.close()
                                         
