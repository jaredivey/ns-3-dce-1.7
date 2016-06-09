import subprocess
import os
import signal
import shlex
import time

import sys
from subprocess import PIPE, Popen

benchmarks = [
"/home/jaredivey/repos/dce-python-sdn/source/ns-3-dce/build/bin_dce/matrixmult_gcc" ,
"/home/jaredivey/repos/dce-python-sdn/build/bin_dce/matrixmult_clang" ,
"python matrixmult.py",
"java MatrixMult",
"java -Xint MatrixMult",
"/home/jaredivey/repos/dce-python-sdn/source/ns-3-dce/build/bin_dce/pidigits_gcc" ,
"/home/jaredivey/repos/dce-python-sdn/build/bin_dce/pidigits_clang" ,
"python pidigits.py",
"java PiDigits",
"java -Xint PiDigits",
"/home/jaredivey/repos/dce-python-sdn/source/ns-3-dce/build/bin_dce/threadtest_gcc" ,
"/home/jaredivey/repos/dce-python-sdn/build/bin_dce/threadtest_clang" ,
"python threadtest.py",
"java ThreadTest",
"java -Xint ThreadTest",
]
#  'PIDIGITS_GCC','PIDIGITS_JAVA','PIDIGITS_CLANG','PIDIGITS_PYTHON',
#  'THREADTEST_GCC','THREADTEST_CLANG','THREADTEST_PYTHON','THREADTEST_JAVA',
#  'PING_GCC','PING_CLANG','PING_PYTHON','PING_JAVA']
        
#file2 is the program output    
file2 = "../simulationResult/wns3-benchmark-ext-output-final.txt"
#clear the file
out2 = open(file2,'wb')
out2.close();
out2 = open(file2,'ab')  

file3 = "../simulationResult/wns3-benchmark-ext-resource-final.txt"
#clear the file
out3 = open(file3,'wb')
out3.close();
out3 = open(file3,'ab')

file4 = "../simulationResult/wns3-benchmark-ext-result-final.txt"
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

for benchmark in range(0,15):
    for iteration in range(0,10):
        start_time = time.time ()
        out3.write("processing benchmark = {0}, iteration = {1}, at {2}\n".format(benchmarks[benchmark],iteration,time.strftime('%Y-%m-%d %X',time.localtime(time.time()))))
        out3.flush()
        out4.write(str(benchmarks[benchmark])+ " ")
        out4.flush()

        print "processing benchmark = {0}, iteration = {1}, at {2}".format(benchmarks[benchmark],iteration,time.strftime('%Y-%m-%d %X',time.localtime(time.time())))
        cmd1 = benchmarks[benchmark]
        print cmd1
        p1 = subprocess.Popen(shlex.split(cmd1),stdout=out2)
        while True:
            try:
                pid_pox = subprocess.check_output(["pgrep", "-f", benchmarks[benchmark]])
                #pid_pox=subprocess.check_output(["./GetPid", str(langChoice), str(loopCount), str(iteration)])
                break
            except subprocess.CalledProcessError:
                time.sleep(1)
                pass
        print pid_pox
        #out4.write(str(langChoice)+ " " + str(loopCount) + " " + str(pid_pox))
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
                if line.find("_gcc") != -1 or line.find("_clang") != -1 or line.find("python") != -1 or line.find("java") != -1:                   
                    temp_private = ToFloat(line,1)
                    temp_share = ToFloat(line,4)
                    temp_total = ToFloat(line,7)
                    if temp_total > max_total:
                        max_total = temp_total
                        max_private = temp_private
                        max_share = temp_share           
            print "temp_total = {0}, max_total = {1} \n".format(temp_total, max_total)
            try:
                pid_pox = subprocess.check_output(["pgrep", "-f", benchmarks[benchmark]])
                #pid_pox=subprocess.check_output(["./GetPid", str(langChoice), str(loopCount), str(iteration)])
            except subprocess.CalledProcessError:
                out4.write( "Duration: {0}\n".format(str(time.time() - start_time)) )
                out4.flush()
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
                                         
