# About this Project
evo-sim is a framework built for simulating the behavior asynchronous evolutionary algorithms incredibly quickly, particularly in the pursuit of an understanding of evaluation time bias.

# GECCO 2025 Reproduction
There are three binaries you must run after building the project (see below) to reproduce the results seen in the paper:

```
# To run the experiments
build/src/repro         # Reproduction experiments
build/src/modulate      # Crossover / weight initialization experiments.
build/src/mitigation    # Mitigation experiments
```

These scripts will place their results in the `results` folder. To generate the plots as seen in the paper, first setup the python environment:

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Then run the scripts:

```
python3 plot/plot_reproduction.py
python3 plot/plot_wall_clock.py
python3 plot/plot_crossover_initialization.py
python3 plot/plot_mitigation.py
```

Which will place the generated figures as PDFs in the `figures` directory.

# Build Instructions
This project makes use of the C++ module system and therefore you must use a new C++ compiler. Clang version 19.1.6 is confirmed to work on mac.

So, create the build directory: 
```
mkdir build
```

Then use cmake to generate the build files:
```
cd build/
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
```

If you have multiple compilers installed, you may have to point cmake to the correct one. For example, this is what must be done on a mac to avoid conflicts with Apple clang:
```
cmake -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ -DCMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES=/opt/homebrew/include -GNinja -DCMAKE_BUILD_TYPE=Release ..
```

Then, just run `ninja` to build:
```
ninja
```
