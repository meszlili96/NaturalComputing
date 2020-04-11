## How to make the video

1. run-ObstaclesGrid.js and run-ObstaclesSimple.js are both configured to simulation as a set of images.
2. run `node run-ObstaclesGrid.js`, its output will be in `output\img\ObstaclesGrid`
3. Images can be converted to video with `ffmpeg` library:
```
ffmpeg -r 60 -f image2 -i output/img/ObstaclesGrid/Obstacles-t%d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p output/mp4/ObstaclesGrid.mp4
```
4. As far as I understand with `RUNTIME` parameter in `simsettings`  we can change video duration
