import scipy.spatial
import cv2 as cv
import numpy as np
import matplotlib
#matplotlib.rcParams.update({'font.size': 22})
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import linear_model, datasets

def basic(img):
    img = cv.medianBlur(img,5)
    ret,th1 = cv.threshold(img,50,255,cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
                cv.THRESH_BINARY,11,2)
    th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv.THRESH_BINARY,11,2)
    titles = ['Original Image', 'Global Thresholding w/ blur (v = 50)',
                'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.figure()
    return th1

def otsu(img):
    #return _binarizeStarfieldImage(img)
    # global thresholding
    #img = cv.fastNlMeansDenoising(5,None,7,21)
    ret1,th1 = cv.threshold(img,50,255,cv.THRESH_BINARY)
    # Otsu's thresholding
    ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # plot all the images and their histograms
    images = [img, th1,
              img, th2]
              #blur, 0, th3]
    titles = ['Original Noisy Image',#'Histogram',
                'Global Thresholding (v=50)',
              'Original Noisy Image',#'Histogram',
                "Otsu's Thresholding"]
              #'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
    '''for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.figure()'''
    return th2
#    for i in range(2):
#        plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
#        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
#        #plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
#        #plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
#        plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
#        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
#    plt.show()

def add_drift(img, mean, variance):
    pass

def add_sensor_noise(img, mean=0, sigma=0.40):
    # http://www.awaiba.com/product/naneye/ 1.2 dn
    # hubble 1x1 binning Amp-D ~1.9 sigma read noise
    # WISE is about 3
    rv = img.copy()
    noise = img.copy()
    cv.randn(noise, mean, sigma)
    return rv + noise


def _binarizeStarfieldImage(imgray, fudge=20):
    """
    
    :param imgray: grayscale image as returned by cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    :rtype: binary image where stars are retained
    """
    maxThreshold = 150
    
    # Find the average background brightness of the starfield.
    # In most cases this corresponds to the first spike of the image histogram,
    # assuming that the starfield background is the darkest part of the image.
    # The threshold for binarization is then the average plus an empirical value.
    hist = cv.calcHist([imgray],[0],None,[256],[0,255]).reshape(256)
    # Sometimes there are small bumps before the first main spike.
    # To prevent choosing the wrong spike, the histogram is slightly smoothed.
    hist[1:-1] = (hist[:-2] + hist[1:-1] + hist[2:]) / 3 # smoothing with window=3
    histDiff = hist[1:] - hist[:-1]
    firstSpike = np.argmax(histDiff<0)
    threshold = min(firstSpike + fudge, maxThreshold)
        
    _,binary = cv.threshold(imgray, threshold, 255, cv.THRESH_BINARY)
    return binary


def smear(img, count=10):
    import random, math
    rows, cols = img.shape
    #length = random.randrange(200, rows + cols) # in pixels
    length = math.sqrt(rows ** 2 + cols ** 2)
    smear_width = 5 # pixels
    x_start = 0#random.randrange(0, cols)
    y_start = random.randrange(0, rows)
    #angle = random.uniform(0, 2 * 3.1415)
    #x_end = int(x_start + length * math.cos(angle))
    #y_end = int(y_start + length * math.sin(angle))
    x_end = cols#random.randrange(cols / 2, cols)
    y_end = random.randrange(0, rows)

    props = {}
    props['slope'] = (y_end - y_start) / (x_end - x_start)
    props['intercept'] = y_start - x_start * props['slope']
    props['start'] = (x_start, y_start)
    props['end'] = (x_end, y_end)

    #x_end = int(1/slope * length)
    #y_end = int(slope * length)

    #cv.line(img,(x_start,y_start),(x_end ,y_end),(0,0,0),smear_width)
    steps = np.linspace(0, length, count)
    frames = []
    plt.figure()
    xs = []
    ys = []
    for step in steps:
        frame = img.copy()
        x_pos = int(x_start + step / length * (x_end - x_start))
        y_pos = int(y_start + step / length * (y_end - y_start))
        xs.append(x_pos)
        ys.append(y_pos)
        cv.circle(frame, (x_pos, y_pos), smear_width, (0, 0, 0), -1)
        frames.append(frame)
        #plt.imshow(frame,'gray')
        #plt.figure()
        #print((x_pos, y_pos))
    plt.title("Reference Image with Transiting NEO Smear")
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.imshow(frame,'gray')
    plt.figure()
    plt.title("NEO Positions During Image Captures")
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.ylim(rows, 0)
    plt.xlim(0, cols)
    plt.scatter(xs, ys)

    return frames, props
 
def compute_missing_pixels(ref_img, smear_img, row, col):
    pass

def invert(img):
    cv.bitwise_not(img)

def find_anomolies(before, after):
    anomolies = []
    before_bin = otsu(before)
    after_bin = otsu(after)
    #before_bin = before
    #after_bin = after
#    for row in range(before_bin.shape[0]):
#        for col in range(before_bin.shape[1]):
#            if before_bin[row][col] > after_bin[row][col]:
#                anomolies.append([col, row])
    anomolies = np.argwhere(before_bin > after_bin)
    # anomolies comes back with empty lists...
    #anomolies = list(filter(any, anomolies))
    # anoms returned in row,column, but we want them as x,y
    anomolies = [np.flip(a) for a in anomolies if all(a)]
    print('anoms', anomolies)
    return anomolies

def find_stars(frame):
    frame_bin = otsu(frame)
    #frame_bin = frame
#    stars = []
#    for row in range(frame_bin.shape[0]):
#        for col in range(frame_bin.shape[1]):
#            if frame_bin[row][col]:
#                stars.append([row, col])
    stars = cv.findNonZero(frame_bin)
    print('Found {} star pixels out of {} pixels in ref img'.format(len(stars), frame.size))
    return stars

    



def find_smears(frames):
    import itertools, time
    start = time.time()
    stars = find_stars(frames[0])
    num_stars = len(stars)
    all_anomolies = []
    per_frame_times = []
    for i in range(1, len(frames)):
        # O(n^2), but doable by ASIC
        anomolies = find_anomolies(frames[0], frames[i])
        if not anomolies:
            continue
        all_anomolies += anomolies
        print('Found {} anomolies in frame {}'.format(len(anomolies), i))
    #all_anomolies = list(filter(None, all_anomolies))
    #print('anoms:', all_anomolies)

    x, y = zip(*all_anomolies)
    #ransac = linear_model.RANSACRegressor(max_trials=2*num_stars, stop_probability=0.99999, min_samples=len(frames))
    ransac = linear_model.RANSACRegressor()
    ransac.fit(np.asarray(x).reshape(-1, 1), np.asarray(y).reshape(-1, 1))

    coef = ransac.estimator_.coef_
    line_info = {}
    [[line_info['slope']]] = ransac.estimator_.coef_
    [line_info['intercept']] = ransac.estimator_.intercept_


    plt.figure()
    plt.title('Starfield Anomolies')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.xlim(0, frames[0].shape[1])
    plt.ylim(frames[0].shape[0], 0)
    plt.scatter(x, y)
        #cv.circle(frames[0], (anom[0], anom[1]), 10, (0, 0, 0), -1)
    per_frame_time = (time.time() - start) / len(frames)
    print('mean {}s per frame'.format(per_frame_time))
    return line_info

def optical_flow(ref_img, smear_img):
    #star_pos = [(row, col) for row, col in (range(ref_img.shape[0]), range(ref_img.shape[1]))]
    star_map = []
    for row in range(ref_img.shape[0]):
        for col in range(ref_img.shape[1]):
            if ref_img[row][col]:
                star_map.append((row, col))
    star_map = np.ndarray(star_map)
    
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    p1, st, err = cv.calcOpticalFlowPyrLK(ref_img, smear_img, star_map, None, **lk_params)


def graph_props(img, actual, estimate):
    #cv.line(img,actual['start'],actual['end'],(255,0,0),5)

    x_start, x_end = (actual['start'][0], actual['end'][0])
    y_start = int(estimate['slope'] * x_start + estimate['intercept'])
    y_end = int(estimate['slope'] * x_end + estimate['intercept'])

    #cv.line(img,(x_start, y_start), (x_end, y_end), (127,0,0),5)
    
    plt.figure()
    estimate_line, = plt.plot([x_start, x_end], [y_start, y_end], 'g-')
    actual_line, = plt.plot([x_start, x_end], [actual['start'][1], actual['end'][1]], 'r-')
    plt.title("Actual vs. Estimated NEO Trajectory")
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.legend([actual_line, estimate_line], ['Actual Trajectory', 'Estimated Trajectory'])
    plt.imshow(img, 'gray')

def discretize_line(slope, intercept, steps=50):
    return [slope * x + intercept for x in range(steps)]

def line_diff(l0, l1):
    d0 = discretize_line(l0['slope'], l0['intercept'])
    d1 = discretize_line(l1['slope'], l1['intercept'])
    res = []
    for i in range(len(d0)):
        res.append(d1[i] - d0[i])
    return res 

def get_variance():
    import mock
    plt = mock.Mock()
    values = []
    skipped = 0
    for i in range(25):
        base_img = cv.imread('pics/night_sky4.jpg',0)
        frames, props = smear(base_img)
        otsu_img = otsu(base_img)
        frames = [add_sensor_noise(f) for f in frames]
        try:
            line_props = find_smears([base_img] + frames)
        except:
            continue
        print('estimated props', line_props)
        print('actual props:', props)
        diff = line_diff(props, line_props)
        print('mean diff per point', sum(diff) / len(diff))
        import math
        if abs(sum(diff) / len(diff)) > 20:
            print('SKIPPING SAMPLE, total skipped: ', skipped)
            skipped += 1
            continue
        values.append(
            diff
        )
    print('stddev', np.std(values))

def main():
    base_img = cv.imread('pics/night_sky4.jpg',0)
    frames, props = smear(base_img)
    otsu_img = otsu(base_img)
    plt.figure()
    plt.title('Starfield after Otsu Binarization')
    plt.xlabel('X Pixels')
    plt.ylabel('Y Pixels')
    plt.imshow(otsu_img, 'gray')
    frames = [add_sensor_noise(f) for f in frames]
    plt.figure()
    plt.imshow(otsu(frames[1]), 'gray')
    line_props = find_smears([base_img] + frames)
    print('estimated props', line_props)
    print('actual props:', props)
    graph_props(base_img, props, line_props)
    #global_img = basic(img)
    #otsu_img = otsu(img)
    #find_smear(base_img, smear_img)
    #optical_flow(base_img, smear_img)
    plt.show()

def time():
    import mock
    plt = mock.Mock()
    base_img = cv.imread('pics/night_sky4.jpg',0)
    frames, props = smear(base_img)
    otsu_img = otsu(base_img)
    plt.figure()
    plt.title('Starfield after Otsu Binarization')
    plt.xlabel('X Pixels')
    plt.ylabel('Y Pixels')
    plt.imshow(otsu_img, 'gray')
    frames = [add_sensor_noise(f) for f in frames]
    plt.figure()
    plt.imshow(otsu(frames[1]), 'gray')
    line_props = find_smears([base_img] + frames)
    print('estimated props', line_props)
    print('actual props:', props)
    graph_props(base_img, props, line_props)
    #global_img = basic(img)
    #otsu_img = otsu(img)
    #find_smear(base_img, smear_img)
    #optical_flow(base_img, smear_img)
    plt.show()

main()
#get_variance()
