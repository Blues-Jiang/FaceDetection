#Makefile of Opencv

#Cross Compiling on Raspberry Pi
#CC:=arm-linux-gnueabihf-g++
CC:=g++

#OpenCV Libs
CFLAGS := `pkg-config opencv --cflags`
LIBS := `pkg-config opencv --libs`

PROG := main
OBJS := $(PROG).o

all:	$(PROG)

$(PROG):	$(OBJS)
	$(CC) -o $(PROG) $(CFLAGS) $(LIBS) $(OBJS) -g

%.o:	%.cpp
	$(CC) -c $(CFLAGS) $(LIBS) $<

clean:
	rm -f $(OBJS) $(PROG) *.o
