# CUDA 2D CDF construction

See: https://maxliani.wordpress.com/2024/03/09/about-fast-2d-cdf-construction/


## Setup

`git clone --recurse-submodules https://github.com/tcantenot/Cuda2DCDF.git .`

### Visual Studio

`cmake CMakeLists.txt -G "Visual Studio 17 2022" -B visualstudio`

Then open the solution in the folder `visualstudio`.


## Output

The app outputs `conditional_cdf.exr` and `marginal_cdf.exr` in the folder `bin/<Config>` corresponding to the hdri in `data`.