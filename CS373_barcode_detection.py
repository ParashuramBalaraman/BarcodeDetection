# Built in packages
import math
import sys
from pathlib import Path

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# You can add your own functions here:
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):

    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    w = 0
    h = 0
    while h < image_height:
        while w < image_width:
            greyscale = .299 * pixel_array_r[h][w] + .587 * pixel_array_g[h][w] + .114 * pixel_array_b[h][w]
            greyscale_pixel_array[h][w] = round(greyscale)
            w += 1
        w = 0
        h += 1
    return greyscale_pixel_array

def computeVerticalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    gpa = createInitializedGreyscalePixelArray(image_width, image_height)
    w = 0
    h = 0
    while h < image_height:
        while w < image_width:
            gpa[h][w] = 0.0
            w += 1
        w = 0
        h += 1
    w = 1
    h = 1
    while h < image_height - 1:
        while w < image_width - 1:
            left = (pixel_array[h - 1][w - 1] * -1) + (pixel_array[h][w - 1] * -2) + (pixel_array[h + 1][w - 1] * -1)
            right = (pixel_array[h - 1][w + 1] * 1) + (pixel_array[h][w + 1] * 2) + (pixel_array[h + 1][w + 1] * 1)
            value = float((left + right) / 8)
            if value < 0:
                value = value * -1
            gpa[h][w] = value
            w += 1
        w = 1
        h += 1
    return gpa

def computeHorizontalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    gpa = createInitializedGreyscalePixelArray(image_width, image_height)
    w = 0
    h = 0
    while h < image_height:
        while w < image_width:
            gpa[h][w] = 0.0
            w += 1
        w = 0
        h += 1
    w = 1
    h = 1
    while h < image_height - 1:
        while w < image_width - 1:
            down = (pixel_array[h + 1][w - 1] * -1) + (pixel_array[h + 1][w] * -2) + (pixel_array[h + 1][w + 1] * -1)
            up = (pixel_array[h - 1][w - 1] * 1) + (pixel_array[h - 1][w] * 2) + (pixel_array[h - 1][w + 1] * 1)
            value = float((up + down) / 8)
            if value < 0:
                value = value * -1
            gpa[h][w] = value
            w += 1
        w = 1
        h += 1
    return gpa

def computeAbsoluteGradient(vertical_gradient, horizontal_gradient, image_width, image_height):
    w = 0
    h = 0
    while h < image_height:
        while w < image_width:
            diff = vertical_gradient[h][w] - horizontal_gradient[h][w]
            if diff < 0:
                vertical_gradient[h][w] = (diff * -1)
            else:
                vertical_gradient[h][w] = diff
            w += 1
        h += 1
        w = 0
    return vertical_gradient

def applyGaussianFilter(pixel_array, image_width, image_height):
    nwidth = image_width + 2
    nheight = image_height + 2
    gpa = createInitializedGreyscalePixelArray(nwidth, nheight)
    w = 0
    h = 0
    w2 = 1
    h2 = 1
    while h < image_height:
        while w < image_width:
            gpa[h2][w2] = pixel_array[h][w]
            w += 1
            w2 += 1
        w = 0
        w2 = 1
        h += 1
        h2 += 1
    gpa[0][0] = pixel_array[0][0]
    gpa[0][1] = pixel_array[0][0]
    gpa[1][0] = pixel_array[0][0]
    gpa[0][nwidth - 1] = pixel_array[0][image_width - 1]
    gpa[0][nwidth - 2] = pixel_array[0][image_width - 1]
    gpa[1][nwidth - 1] = pixel_array[0][image_width - 1]
    gpa[nheight - 1][0] = pixel_array[image_height - 1][0]
    gpa[nheight - 2][0] = pixel_array[image_height - 1][0]
    gpa[nheight - 1][1] = pixel_array[image_height - 1][0]
    gpa[nheight - 1][nwidth - 1] = pixel_array[image_height - 1][image_width - 1]
    gpa[nheight - 1][nwidth - 2] = pixel_array[image_height - 1][image_width - 1]
    gpa[nheight - 2][nwidth - 1] = pixel_array[image_height - 1][image_width - 1]

    w = 1
    w2 = 2
    while w < image_width - 1:
        gpa[0][w2] = pixel_array[0][w]
        gpa[nheight - 1][w2] = pixel_array[image_height - 1][w]
        w += 1
        w2 += 1
    h = 1
    h2 = 2
    while h < image_height - 1:
        gpa[h2][0] = pixel_array[h][0]
        gpa[h2][nwidth - 1] = pixel_array[h][image_width - 1]
        h += 1
        h2 += 1

    gpa2 = createInitializedGreyscalePixelArray(image_width, image_height)
    w = 1
    h = 1
    w2 = 0
    h2 = 0
    while h < nheight - 1:
        while w < nwidth - 1:
            top = gpa[h - 1][w - 1] + (gpa[h - 1][w] * 2) + gpa[h - 1][w + 1]
            mid = (gpa[h][w - 1] * 2) + (gpa[h][w] * 4) + (gpa[h][w + 1] * 2)
            bottom = gpa[h + 1][w - 1] + (gpa[h + 1][w] * 2) + gpa[h + 1][w + 1]
            value = float(top + bottom + mid) / 16
            gpa2[h2][w2] = value
            w += 1
            w2 += 1
        w = 1
        w2 = 0
        h += 1
        h2 += 1

    return gpa2

def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    ge = createInitializedGreyscalePixelArray(image_width, image_height)
    w = 0
    h = 0
    while h < image_height:
        row = pixel_array[h]
        while w < image_width:
            digit = row[w]
            if digit >= threshold_value:
                ge[h][w] = 255
            w += 1
        w = 0
        h += 1
    return ge

def computeDilation(pixel_array, image_width, image_height):
    nwidth = image_width + 2
    nheight = image_height + 2
    gpa = createInitializedGreyscalePixelArray(nwidth, nheight)
    w = 0
    h = 0
    w2 = 1
    h2 = 1
    while h < image_height:
        while w < image_width:
            gpa[h2][w2] = pixel_array[h][w]
            w += 1
            w2 += 1
        w = 0
        w2 = 1
        h += 1
        h2 += 1
    w = 0
    h = 0
    while h < image_height:
        row = pixel_array[h]
        while w < image_width:
            kernel = row[w]
            if kernel > 0:
                gpa[h][w] = 1
                gpa[h][w + 1] = 1
                gpa[h][w + 2] = 1
                gpa[h + 1][w] = 1
                gpa[h + 1][w + 1] = 1
                gpa[h + 1][w + 2] = 1
                gpa[h + 2][w] = 1
                gpa[h + 2][w + 1] = 1
                gpa[h + 2][w + 2] = 1
            w += 1
        h += 1
        w = 0
    gpa2 = createInitializedGreyscalePixelArray(image_width, image_height)
    w = 0
    h = 0
    w2 = 1
    h2 = 1
    while h2 < nheight - 1:
        while w2 < nwidth - 1:
            gpa2[h][w] = gpa[h2][w2]
            w += 1
            w2 += 1
        h += 1
        h2 += 1
        w = 0
        w2 = 1
    return gpa2

def computeErosion(pixel_array, image_width, image_height):
    nwidth = image_width + 2
    nheight = image_height + 2
    gpa = createInitializedGreyscalePixelArray(nwidth, nheight)
    w = 0
    h = 0
    w2 = 1
    h2 = 1
    while h < image_height:
        while w < image_width:
            gpa[h2][w2] = pixel_array[h][w]
            w += 1
            w2 += 1
        w = 0
        w2 = 1
        h += 1
        h2 += 1
    w = 0
    h = 0
    while h < image_height:
        row = pixel_array[h]
        while w < image_width:
            kernel = row[w]
            if kernel == 0:
                gpa[h][w] = 0
                gpa[h][w + 1] = 0
                gpa[h][w + 2] = 0
                gpa[h + 1][w] = 0
                gpa[h + 1][w + 1] = 0
                gpa[h + 1][w + 2] = 0
                gpa[h + 2][w] = 0
                gpa[h + 2][w + 1] = 0
                gpa[h + 2][w + 2] = 0
            w += 1
        h += 1
        w = 0
    gpa2 = createInitializedGreyscalePixelArray(image_width, image_height)
    w = 0
    h = 0
    w2 = 1
    h2 = 1
    while h2 < nheight - 1:
        while w2 < nwidth - 1:
            if gpa[h2][w2] > 1:
                gpa2[h][w] = 1
            else:
                gpa2[h][w] = gpa[h2][w2]
            w += 1
            w2 += 1
        h += 1
        h2 += 1
        w = 0
        w2 = 1
    w = 0
    row = gpa2[0]
    row1 = gpa2[image_height - 1]
    while w < image_width:
        row[w] = 0
        row1[w] = 0
        w += 1
    h = 0
    while h < image_height:
        gpa2[h][0] = 0
        gpa2[h][image_width - 1] = 0
        h += 1
    return gpa2

def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    gpa = createInitializedGreyscalePixelArray(image_width, image_height)
    label = 1
    keys = {}
    w = 0
    h = 0
    w2 = 1
    h2 = 1
    while h < image_height:
        while w < image_width:
            if pixel_array[h][w] > 0:
                gpa[h][w] = label
                pixel_array[h][w] = 0
                queue = Queue()
                queue.enqueue((h,w))
                count = 1
                while queue.isEmpty() != True:
                    tup = queue.dequeue()
                    h3 = tup[0]
                    w3 = tup[1]
                    if h3 != 0 and pixel_array[h3 - 1][w3] > 0:
                        gpa[h3 - 1][w3] = label
                        queue.enqueue((h3 - 1, w3))
                        pixel_array[h3 - 1][w3] = 0
                        count += 1
                    if w3 != 0 and pixel_array[h3][w3 - 1] > 0:
                        gpa[h3][w3 - 1] = label
                        queue.enqueue((h3, w3 - 1))
                        pixel_array[h3][w3 - 1] = 0
                        count += 1
                    if w3 != image_width - 1 and pixel_array[h3][w3 + 1] > 0:
                        gpa[h3][w3 + 1] = label
                        queue.enqueue((h3, w3 + 1))
                        pixel_array[h3][w3 + 1] = 0
                        count += 1
                    if h3 != image_height - 1 and pixel_array[h3 + 1][w3] > 0:
                        gpa[h3 + 1][w3] = label
                        queue.enqueue((h3 + 1, w3))
                        pixel_array[h3 + 1][w3] = 0
                        count += 1
                keys[label] = count
                label+=1
            w2+=1
            w+=1
        h+=1
        h2+=1
        w = 0
        w2 = 1
    return (gpa, keys)

def findConnectedComponent(pixel_array, size_dict, image_width, image_height):
    dict_values = []
    for sz in size_dict.keys():
        dict_values.append((sz, size_dict[sz]))
    descend_size = [dict_values[0]]
    x = 1
    while x < len(dict_values):
        i = 0
        length = len(descend_size)
        while i < length:
            if dict_values[x][1] > descend_size[i][1]:
                descend_size.insert(i, dict_values[x])
                i = length
            if i == length - 1:
                descend_size.append(dict_values[x])
            i += 1
        x += 1
    for val in descend_size:
        min_y = 10000000
        max_y = 0
        min_x = 10000000
        max_x = 0
        w = 0
        h = 0
        while h < image_height:
            while w < image_width:
                if pixel_array[h][w] == val[0]:
                    if h < min_y:
                        min_y = h
                    if h > max_y:
                        max_y = h
                    if w < min_x:
                        min_x = w
                    if w > max_x:
                        max_x = w
                w += 1
            h += 1
            w = 0
        x_diff = max_x - min_x
        y_diff = max_y - min_y
        aspect_ratio = 0
        if x_diff > y_diff:
            aspect_ratio = x_diff / y_diff
        else:
            aspect_ratio = y_diff / x_diff
        if aspect_ratio <= 2:
            xy_values = [min_x, min_y, max_x, max_y]
            print(val)
            return xy_values


# This is our code skeleton that performs the barcode detection.
# Feel free to try it on your own images of barcodes, but keep in mind that with our algorithm developed in this assignment,
# we won't detect arbitrary or difficult to detect barcodes!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    filename = "Barcode2"
    input_filename = "images/"+filename+".png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(filename+"_output.png")
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')


    # STUDENT IMPLEMENTATION here
    #1
    pixel_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    #2
    vertical_gradient = computeVerticalEdgesSobelAbsolute(pixel_array, image_width, image_height)
    horizontal_gradient = computeHorizontalEdgesSobelAbsolute(pixel_array, image_width, image_height)
    absolute_gradient = computeAbsoluteGradient(vertical_gradient, horizontal_gradient, image_width, image_height)
    #3
    pixel_array = applyGaussianFilter(absolute_gradient, image_width, image_height)
    pixel_array = applyGaussianFilter(pixel_array, image_width, image_height)
    pixel_array = applyGaussianFilter(pixel_array, image_width, image_height)
    pixel_array = applyGaussianFilter(pixel_array, image_width, image_height)

    #4
    threshold_value = 15
    threshold = computeThresholdGE(pixel_array, threshold_value, image_width, image_height)
    #5
    dilation = computeDilation(threshold, image_width, image_height)
    dilation = computeDilation(dilation, image_width, image_height)
    erosion = computeErosion(dilation, image_width, image_height)
    erosion = computeErosion(erosion, image_width, image_height)
    erosion = computeErosion(erosion, image_width, image_height)
    pixel_array = computeErosion(erosion, image_width, image_height)
    #6&7
    (pixel_array, size) = computeConnectedComponentLabeling(pixel_array, image_width, image_height)
    xy_values = findConnectedComponent(pixel_array, size, image_width, image_height)


    
    px_array = px_array_r

    # Compute a dummy bounding box centered in the middle of the input image, and with as size of half of width and height
    # Change these values based on the detected barcode region from your algorithm
    center_x = (xy_values[0] + xy_values[2]) / 2.0
    center_y = (xy_values[1] + xy_values[3]) / 2.0
    bbox_min_x = xy_values[0]
    bbox_max_x = xy_values[2]
    bbox_min_y = xy_values[1]
    bbox_max_y = xy_values[3]

    # The following code is used to plot the bounding box and generate an output for marking
    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()