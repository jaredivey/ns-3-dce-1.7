all: server_clang client_clang server_client_clang matrixmult_clang pidigits_clang threadtest_clang 

server_clang: server.o
	clang -o /home/jaredivey/repos/dce-python-sdn/build/bin_dce/server_clang -pie -rdynamic server.o

server.o: server.c
	clang -fPIC -c server.c

client_clang: client.o
	clang -o /home/jaredivey/repos/dce-python-sdn/build/bin_dce/client_clang -pie -rdynamic client.o

client.o: client.c
	clang -fPIC -c client.c

server_client_clang: server_client.o
	clang -o /home/jaredivey/repos/dce-python-sdn/build/bin_dce/server_client_clang -pie -rdynamic server_client.o

server_client.o: server_client.c
	clang -fPIC -c server_client.c

matrixmult_clang: matrixmult.o
	clang -o /home/jaredivey/repos/dce-python-sdn/build/bin_dce/matrixmult_clang -pie -rdynamic matrixmult.o

matrixmult.o: matrixmult.c
	clang -fPIC -c matrixmult.c

pidigits_clang: pidigits.o
	clang -o /home/jaredivey/repos/dce-python-sdn/build/bin_dce/pidigits_clang -pie -rdynamic pidigits.o

pidigits.o: pidigits.c
	clang -fPIC -c pidigits.c

threadtest_clang: threadtest.o
	clang -o /home/jaredivey/repos/dce-python-sdn/build/bin_dce/threadtest_clang -pie -rdynamic threadtest.o -lpthread

threadtest.o: threadtest.c
	clang -fPIC -c threadtest.c

#ping_clang: ping.o
#	clang -o /home/jaredivey/repos/dce-python-sdn/build/bin_dce/ping_clang -pie -rdynamic ping.o

#ping.o: ping.c
#	clang -O -fPIC -c ping.c

clean:
	rm -f server.o client.o server_client.o matrixmult.o pidigits.o threadtest.o
