## Usage

```text
usage: epi-depth <images-dir> <chess-x> <chess-y>
                 <scale-factor> <is-moving-to-right> <min-deg> <max-deg> <deg-step>
                 <depth-scale> <depth-offset>

 images-dir           directory of images
                      images used for calibration should be put in 'calib'
                      images used for generate depth map should be put in 'imgs' sub dir
 chess-x, chess-y     size of chessboard used for calibration
 scale-factor         scale factor that will be applied to all input images
 is-moving-to-right   0 if camera moves to left, 1 if camera moves to right
 min-deg              min search degree from horizon
 max-deg              max search degree from horizon
 deg-step             search step of degree
 depth-scale          scale of 0-1 depth when generate point cloud
 depth-offset         offset of scaled depth when generate point cloud
```

## Build

OpenCV 4.x is needed (can be installed with vcpkg).

```text
mkdir build
cd build
cmake ..
cmake --build .
```

