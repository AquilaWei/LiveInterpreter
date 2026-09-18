# Icons

`icon.png` (512x512 RGBA) is the source. The rest are generated from it:

```bash
magick icon.png -resize 32x32   -define png:color-type=6 PNG32:32x32.png
magick icon.png -resize 128x128 -define png:color-type=6 PNG32:128x128.png
magick icon.png -resize 256x256 -define png:color-type=6 PNG32:128x128@2x.png
magick icon.png -define icon:auto-resize=128,64,48,32,16 icon.ico
```

`PNG32:` is not optional -- ImageMagick will happily hand back a palette PNG,
and the bundler wants RGBA. `icon.ico` is for a Windows package;
the Linux bundler ignores it.
