.SUFFIXES: .c .o .f .F

##CC			=  gcc
CC			=  mpicc
CFLAGS			= -g -Wall -fno-omit-frame-pointer -fsanitize=bounds
##LIBS			=  -L/project/scicom/scicom00/SOFT/lib/linux/ -lblas
LIBS			=  -lblas 

FILES =  main.o auxil1.o 

main.ex: $(FILES) 
	${CC} ${CFLAGS} -o main.ex -lm $(FILES) $(LIBS)

all: main.ex

.c.o:
	${CC} ${CFLAGS} $< -c -o $@ $(LIBS)

clean:
	rm *.o *.ex
