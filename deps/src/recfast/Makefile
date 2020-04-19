#### Makefile ####
## library ##
target=librecfast.so

# complier and options
# FC = ifort
FC = gfortran
FLAG = -ffixed-line-length-none -shared -fPIC -g -O0
FLAGS = $(FLAG) $(DBAG)

# directories
libroot = ../../lib/

# files to be compiled
# OBJ = toms760.o inifile.o utils.o spin_alm_tools.o remap.o #HealpixObj.o HealpixVis.o

# $(target): $(OBJ)
# 	ar crs $@ $^

recfast: recfast.for
	$(FC) $(FLAGS) $(OPTION) recfast.for -o $(target)

install:
	mv *.so $(libroot)

clean:
	rm -f *.o* *.mod *.a *.so
