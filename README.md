# HLPtool
A progam for finding solutions to the Hex Layer Problem. 

# Usage
First, download the latest release for your system. Note that as of right now, only 64-bit x86 systems with AVX2 are supported. If your computer was made after 2013 you are probably fine. Once you have downloaded it, simply navigate to the folder it is downloaded in and run it with `./hlpsolve` for Linux or `hlpsolve.exe` (in command prompt) for Windows.

To input the function you want solved, type out the desired outputs in order using hexadecimal. For example:

```bash
$ # square root, rounded down:
$ ./hlpsolve 0111222223333333

$ # digits of pi:
$ ./hlpsolve 3141592653589793

$ # rotate top bit to bottom:
$ ./hlpsolve 02468ace13579bdf

```

You can insert spaces at any point, which can help make sure you typed it correctly:

```bash
$ ./hlpsolve 0111 2222 2333 3333
```



## Optimal solutions
The solutions found by this program are almost always the shortest possible length. However, at times it produces a solution a layer or two longer than the true minimum. This can be prevented by passing in `-p` or `--perfect`:

```bash
$ ./hlpsolve 3141 5926 5358 9793 --perfect
```

However, this generally makes it take significantly longer to find a solution, and most of the time it's the same solution it would've found otherwise, which is why this is not the default behaviour.

# Compiling
To compile, make sure you have `autoconf` installed and run the following:

```bash
$ autoreconf
$ ./configure # Linux
$ ./configure --build x86_64-linux-gnu --host x86_64-w64-mingw32 # Windows cross-compile
$ make
```

there's a chance you may need to use `gnulib-tool --import argp malloc-gnu` 
