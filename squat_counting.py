#Download videos from https://drive.google.com/file/d/1TNwCB5o3-joGgclbXkI01f2r4FazMf-w/view?usp=sharing

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter

def squat_processing(all_boundingboxes):
    peaks_arr = []
    n = int(max(all_boundingboxes[:, 1]) + 1)
    plt.figure(figsize=(10, 6))
    for i in range(n):
        idx = np.where(all_boundingboxes[:, 1] == i)[0]
        t_values = all_boundingboxes[idx, 0]
        h_values = all_boundingboxes[idx, 5] 

        percentile = np.percentile(h_values, 75)
        peaks, properties = find_peaks(h_values, height=percentile, distance=28)
        if len(peaks) == 0:
            continue
        peaks_arr.append(len(peaks))

        plt.subplot(n, 1, i + 1)
        plt.plot(t_values, h_values, label="Bounding Box Height")
        plt.plot(t_values[peaks], h_values[peaks], "rx", label=f"Lower Peaks (< {percentile:.2f})")
        plt.xlabel("Time (Frames)")
        plt.ylabel("Height")
        plt.legend()
        plt.title(f"Squat Counting {i} : {len(peaks)}")

    plt.tight_layout()
    plt.show()
    return peaks_arr


def white_processing(img, bg, all_boundingboxes, t):
    diffc = cv2.absdiff(img, bg)
    diffg = cv2.cvtColor(diffc, cv2.COLOR_BGR2GRAY)
    bwmask = cv2.inRange(diffg, 35, 255)
    bwmask_median = cv2.medianBlur(bwmask, 21)
    bwmask_dilated = cv2.dilate(bwmask_median, np.ones((65, 20), np.uint8))
    bwmask_open = cv2.morphologyEx(bwmask_dilated, cv2.MORPH_OPEN, np.ones((5, 20), np.uint8))
    contours, hierarchy = cv2.findContours(bwmask_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    im_out_boundingbox = img.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h / w > 2 and w * h > 1e4:
            # cv2.putText(
            #     im_out_boundingbox,
            #     f"{img.shape[0] / h}",
            #     (x, y),
            #     fontFace=cv2.FONT_HERSHEY_PLAIN,
            #     fontScale=3,
            #     thickness=2,
            #     color=(0, 0, 255),
            # )
            cv2.rectangle(im_out_boundingbox, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if 250 <= t:
                all_boundingboxes.append([t, 0, x, y, w, img.shape[0] / h])
    return im_out_boundingbox
    
def black_processing(img, bg, all_boundingboxes, t):
    diffc = cv2.absdiff(img, bg)
    # eroded_image = cv2.erode(diffc, np.ones((1, 25), np.uint8))
    diffg = cv2.cvtColor(diffc, cv2.COLOR_BGR2GRAY)
    bwmask = cv2.inRange(diffg, 35, 255)
    bwmask_median = cv2.medianBlur(bwmask, 17)
    bwmask_dilated = cv2.dilate(bwmask_median, np.ones((100, 20), np.uint8))
    bwmask_open = cv2.morphologyEx(bwmask_dilated, cv2.MORPH_OPEN, np.ones((5, 20), np.uint8))
    contours, hierarchy = cv2.findContours(bwmask_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    im_out_boundingbox = img.copy()
    index = 1
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h / w > 2 and w * h > 8e3:
            # cv2.putText(
            #     im_out_boundingbox,
            #     f"{h / w}",
            #     (x, y),
            #     fontFace=cv2.FONT_HERSHEY_PLAIN,
            #     fontScale=3,
            #     thickness=2,
            #     color=(0, 0, 255),
            # )
            cv2.rectangle(im_out_boundingbox, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if 330 <= t <= 975:
                all_boundingboxes.append([t, index, x, y, w, img.shape[0] / h])
            index += 1
    return im_out_boundingbox

def squatCounting(filename):
    cap = cv2.VideoCapture(filename)
    haveFrame, bg = cap.read()
    w_bg = bg[:, 100:300].copy()
    b_bg = bg[:, 320:500].copy()
    t = 0

    all_boundingboxes = []
    peaks_arr = []

    while cap.isOpened():
        haveFrame, im = cap.read()

        if not haveFrame or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

        im_w_bg = im[:, 100:300].copy()
        im_b_bg = im[:, 320:500].copy()
        im_w = white_processing(im_w_bg, w_bg, all_boundingboxes, t)
        im_b = black_processing(im_b_bg, b_bg, all_boundingboxes, t)
        
        t += 1
        cv2.imshow('image white background', im_w)
        cv2.moveWindow('image white background', 10, 10)
        cv2.imshow('image black background', im_b)
        cv2.moveWindow('image black background', im_w.shape[1], 10)

    all_boundingboxes = np.array(all_boundingboxes)
    all_boundingboxes[:, 5] = (all_boundingboxes[:, 5] - np.min(all_boundingboxes[:, 5])) / (np.max(all_boundingboxes[:, 5]) - np.min(all_boundingboxes[:, 5]))
    n = int(max(all_boundingboxes[:, 1]) + 1)
    try:
        if n == 3:
            idx1 = np.where(all_boundingboxes[:, 1] == 1)[0]
            idx2 = np.where(all_boundingboxes[:, 1] == 2)[0]
            all_boundingboxes[idx1, 1], all_boundingboxes[idx2, 1] = all_boundingboxes[idx2, 1], all_boundingboxes[idx1, 1]
    except Exception as e:
        pass
    
    for i in range(n):
        idx = np.where(all_boundingboxes[:, 1] == i)[0]
        all_boundingboxes[idx, 5] = gaussian_filter(all_boundingboxes[idx, 5], sigma=3.5)
        # print( np.mean(all_boundingboxes[idx, 5]))
        all_boundingboxes[idx, 5] = (all_boundingboxes[idx, 5] >= np.mean(all_boundingboxes[idx, 5]) - 0.01).astype(int)
    
    peaks_arr = squat_processing(all_boundingboxes)
    # print(all_boundingboxes)
    # print(n)

    # for i in range(n):
    #     idx = np.where(all_boundingboxes[:, 1] == i)[0]
    
    #     plt.subplot(n, 1, i + 1)
    #     plt.plot(all_boundingboxes[idx, 0], all_boundingboxes[idx, 5], label="Data")
    #     plt.fill_between(all_boundingboxes[idx, 0], all_boundingboxes[idx, 5], alpha=0.3)q
    
    #     mean_value = np.mean(all_boundingboxes[idx, 5]) - 0.01
    #     plt.axhline(y=mean_value, color='r', linestyle='--', label=f"Mean: {mean_value:.2f}")
    #     plt.xlabel('X-axis Label')  # Replace with appropriate label
    #     plt.ylabel('Y-axis Label')  # Replace with appropriate label
    #     plt.legend()

    # plt.show()

    cap.release()
    cv2.destroyAllWindows()

    return peaks_arr


print(squatCounting('.\SquatCounting\Squat1_8_9.avi')) #Perfect Answer = [8 9]
print(squatCounting('.\SquatCounting\Squat2_16_17.avi')) #Perfect Answer = [16 17]
print(squatCounting('.\SquatCounting\Squat3_11_9_10.avi')) #Perfect Answer = [11 9 10]
