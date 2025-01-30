#!/usr/bin/env python
import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw, ImageFont
import sys

def rgb_to_hex(r, g, b):
    """Converts RGB color values to a hex string."""
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

if __name__ == '__main__':
    image_file_name = sys.argv[1]
    k_value = int(sys.argv[2])

    img = cv.imread(image_file_name)
    Z = img.reshape((-1,3))
    
    # convert to np.float32
    Z = np.float32(Z)
    
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = k_value
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center) # All of the K-means color clusters (NB. This is BGR not RGB!)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    # Create an image of the K-means color cluster palette (RGB Format)
    img = Image.new('RGB', (800, 800), color = 'white')
    number_of_palette_rows_cols = np.uint8(np.ceil(np.sqrt(K)))
    grid_size = int(img.width/number_of_palette_rows_cols)
    color_index = 0
    default_font = ImageFont.load_default(size = 10)
    for row in range(number_of_palette_rows_cols):
        for col in range(number_of_palette_rows_cols):
            if (color_index < K):
                draw = ImageDraw.Draw(img)
                draw.rectangle([(col*grid_size, row*grid_size),(col*grid_size + grid_size,row * grid_size + grid_size)],fill=tuple(center[color_index])[::-1])
                rgb_values = str(tuple(center[color_index])[::-1])
                rgb_text = f'RGB: {rgb_values}'
                # Print for output as p5js color RGB format; e.g. color(255,0,0) 
                # print(f'color{tuple(center[color_index])[::-1]}')                 
                hex_values = str(rgb_to_hex(*tuple(center[color_index])[::-1]))
                hex_text = f'Hex: {hex_values}'
                # Print for output as p5js color HEX format; e.g. color('#ff0000')
                print(f'color("{str(rgb_to_hex(*tuple(center[color_index])[::-1]))}")') 
                draw.text((col*grid_size,row*grid_size),rgb_text,font=default_font,fill=(0,0,0))
                draw.text((col*grid_size,row*grid_size + grid_size/10),hex_text,font=default_font,fill=(0,0,0))
            color_index += 1
    img.show()

    # This shows the K-means clustered image:
    # cv.imshow('res2',res2)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

