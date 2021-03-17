.SUFFIXES: .c .o .f .F

##CC			=  gcc
CCX			=  mpicc
CC_FLAGS			= -std=c++11 -g -Wall
##LIBS			=  -L/project/scicom/scicom00/SOFT/lib/linux/ -lblas
LIBS			=  -lblas 

FILES =  main.o auxil1.o 

main.ex: $(FILES) 
	${CCX} ${CC_FLAGS} -o main.ex -lm $(FILES) $(LIBS)

all: main.ex

.cc.o:
	${CCX} ${CC_FLAGS} $< -c -o $@ $(LIBS)

clean:
	rm *.o *.ex
