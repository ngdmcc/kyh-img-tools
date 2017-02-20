import base64, cv2, os, glob
import numpy as np

def resize_pad_image(img, new_dims, pad_output=True):
    old_height, old_width, ch = img.shape
    old_ar = float(old_width) / float(old_height)
    new_ar = float(new_dims[0]) / float(new_dims[1])
    undistorted_scale_factor = [1.0, 1.0]  # if you want to resize bounding boxes on a padded img you'll need this
    if pad_output is True:
        if new_ar > old_ar:
            new_width = old_height * new_ar
            padding = abs(new_width - old_width)
            img = cv2.copyMakeBorder(img, 0, 0, 0, int(padding), cv2.BORDER_CONSTANT, None, [0, 0, 0])
            undistorted_scale_factor = [float(old_width) / (float(new_dims[1]) * old_ar),
                                        float(old_height) / float(new_dims[1])]
        elif new_ar < old_ar:
            new_height = old_width / new_ar
            padding = abs(new_height - old_height)
            img = cv2.copyMakeBorder(img, 0, int(padding), 0, 0, cv2.BORDER_CONSTANT, None, [0, 0, 0])
            undistorted_scale_factor = [float(old_width) / float(new_dims[0]),
                                        float(old_height) / (float(new_dims[0]) / old_ar)]
        elif new_ar == old_ar:
            scale_factor = float(old_width) / new_dims[0]
            undistorted_scale_factor = [scale_factor, scale_factor]
    outimg = cv2.resize(img, (new_dims[0], new_dims[1]))
    return outimg, undistorted_scale_factor


def squarify_bbox(r):
    """
    :param r: input rect
    :return: square bbox (width==height)
    """
    top_left = (r[0], r[1])
    bottom_right = (r[2], r[3])
    w = bottom_right[0] - top_left[0]
    h = bottom_right[1] - top_left[1]

    if w > h:
        diff = w - h
        pad = diff / 2.0
        pad_rounded = int(round(pad))
        r[1] -= pad_rounded
        r[3] += pad_rounded
    elif w < h:
        diff = h - w
        pad = diff / 2.0
        pad_rounded = int(round(pad))
        r[0] -= pad_rounded
        r[2] += pad_rounded

    for num in r:
        if num < 0:
            # print "negative number in array, continuing..."
            return None
    return [int(r[0]), int(r[1]), int(r[2]), int(r[3])]


def draw_bboxes(im, bboxes, color, label="object"):
    for bb in bboxes:
        draw_bbox(im, bb, color, label=label)

def draw_bbox(im, bb, color, label="object", font_size=1.5, frame_index=0):
    if len(bb) != 0:
        bb = np.array(bb, dtype=np.int)
        cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
        cv2.putText(im, label, (bb[0], bb[1]), 1, font_size, color, 2)
        draw_frame_index(im, frame_index)

def show_image(im, label="image", delay=0):
    cv2.imshow(label, im)
    cv2.waitKey(delay) & 0xFF == ord('q')

def create_empty_dirtree(srcdir, dstdir, onerror=None):
    srcdir = os.path.abspath(srcdir)
    srcdir_prefix = len(srcdir) + len(os.path.sep)
    try:
        os.mkdir(srcdir)
    except OSError as e:
        if onerror is not None:
            onerror(e)
    for root, dirs, files in os.walk(srcdir, onerror=onerror):
        for dirname in dirs:
            dirpath = os.path.join(dstdir, root[srcdir_prefix:], dirname)
            try:
                os.mkdir(dirpath)
            except OSError as e:
                if onerror is not None:
                    onerror(e)

def crop_img(bbox, im):
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    cropped_img = im[y1:y2, x1:x2]
    return cropped_img

def compute_dist(vec1, vec2, mode='cosine'):
    """
    compute the distance between two given vectors.
    :param vec1: np.array vector
    :param vec2: np.array vector
    :param mode: cosine for cosine distance; l2 for l2 norm distance;
    :return: distance of the input mode
    """
    if mode == 'cosine':
        dist = 1 - np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
    elif mode == 'l2':
        dist = np.linalg.norm(vec1 - vec2)
    else:
        dist = None
    return dist

def make_grids_of_images(images_path, image_shape, grid_shape):
    """
    :param image_path_list: string,
    :param image_shape: tuple, size each image will be resized to for display
    :param grid_shape: tuple, shape of image grid (rows,cols)
    :return: list of grid images in numpy array format

    example usage: grids = make_grids_of_images('/Pictures', (64,64),(5,5))

    """
    # get all images from folder
    img_path_glob = glob.iglob(os.path.join(images_path,'*'))
    img_path_list = []
    for ip in img_path_glob:
        print ip
        if ip.endswith('.jpg') or ip.endswith('.jpeg') or ip.endswith('.png'):
            img_path_list.append(ip)
    if len(img_path_list) < 1:
        print 'No images found at {}'.format(images_path)
        return None
    image_grids = []
    #start with black canvas to draw images to
    grid_image = np.zeros(shape=(image_shape[1] * (grid_shape[1]), image_shape[0] * grid_shape[0], 3),
                            dtype=np.uint8)
    cursor_pos = [0, 0]
    for img_path in img_path_list:
        img = cv2.imread(img_path)
        if img is None:
            print 'ERROR: reading {}. skipping.'.format(img_path)
            continue
        img = cv2.resize(img, image_shape)
        # draw image to black canvas
        grid_image[cursor_pos[1]:cursor_pos[1] + image_shape[1], cursor_pos[0]:cursor_pos[0] + image_shape[0]] = img
        cursor_pos[0] += image_shape[0] # increment cursor x position
        if cursor_pos[0] >= grid_shape[0] * image_shape[0]:
            cursor_pos[1] += image_shape[1] #increment cursor y position
            cursor_pos[0] = 0
            if cursor_pos[1] >= grid_shape[1] * image_shape[1]:
                cursor_pos = [0, 0]
                #reset black canvas
                grid_image = np.zeros(shape=(image_shape[1] * (grid_shape[1]), image_shape[0] * grid_shape[0], 3),
                                      dtype=np.uint8)
        image_grids.append(grid_image)

    return image_grids

grids = make_grids_of_images('/home/kyle/Pictures', (256,256), (2,2))
for img in grids:
    cv2.imshow('img',img)
    cv2.waitKey(100)

