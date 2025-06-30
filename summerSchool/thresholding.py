import cv2
import numpy as np
import time

# --- Configuration ---
camera_index   = 0
retry_interval = 1    # seconds between retries when camera isn’t ready
width, height  = 640, 480

# --- Initialize DirectShow capture on Windows ---
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# --- Create windows ---
for name in ['Controls','RefPreview','BlurGray','ColorDiff','Threshold','Output']:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)

# --- Trackbar callback (no-op) ---
def nothing(x): pass

# --- Create trackbars ---
cv2.createTrackbar('GaussKernel','Controls',1,50, nothing)  # odd size
cv2.createTrackbar('GrayLevel','Controls',100,100,nothing)  # percent gray
cv2.createTrackbar('Thresh','Controls',30,255,nothing)      # binary threshold

# Identity kernel (no extra convolution)
identity_kern = np.array([[0,0,0],[0,1,0],[0,0,0]],dtype=np.float32)

# --- State holders ---
ref_frame_raw = None
last_retry    = time.time()

print("Press 'r' to record reference; 'q' to quit.")

while True:
    # --- Retry logic if camera not ready ---
    if not cap.isOpened():
        if time.time()-last_retry >= retry_interval:
            print("Waiting for camera… retrying DirectShow open()")
            cap.release()
            cap = cv2.VideoCapture(camera_index,cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
            last_retry = time.time()
        blank = np.zeros((200,400,3),np.uint8)
        cv2.putText(blank,"Waiting for camera...",(10,100),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.imshow('Output',blank)
        if cv2.waitKey(1)&0xFF==ord('q'): break
        continue

    # --- Grab frame ---
    ret, frame = cap.read()
    if not ret:
        continue

    # --- Read sliders ---
    k = cv2.getTrackbarPos('GaussKernel','Controls')
    ksize    = k+1 if k%2==0 else max(1,k)
    gray_pct = cv2.getTrackbarPos('GrayLevel','Controls')/100.0
    thresh_v = cv2.getTrackbarPos('Thresh','Controls')

    # --- Blur live frame ---
    blurred_live = cv2.GaussianBlur(frame,(ksize,ksize),0)

    # --- Gray + blend for display ---
    gray_live     = cv2.cvtColor(blurred_live,cv2.COLOR_BGR2GRAY)
    gray_bgr_live = cv2.cvtColor(gray_live,cv2.COLOR_GRAY2BGR)
    blur_gray     = cv2.addWeighted(gray_bgr_live,gray_pct,blurred_live,1-gray_pct,0)
    cv2.imshow('BlurGray',blur_gray)

    # --- If ref recorded, re-blur it and diff ---
    if ref_frame_raw is not None:
        # re-blur reference
        blurred_ref = cv2.GaussianBlur(ref_frame_raw,(ksize,ksize),0)
        # color diff
        color_diff = cv2.absdiff(blurred_live,blurred_ref)
        cv2.imshow('ColorDiff',color_diff)
        # prepare mono ref for contours
        gray_ref      = cv2.cvtColor(blurred_ref,cv2.COLOR_BGR2GRAY)
        ref_processed = cv2.filter2D(gray_ref,-1,identity_kern)
        cv2.imshow('RefPreview',ref_processed)

        # find contours on mono diff
        diff_mono   = cv2.absdiff(ref_processed,gray_live)
        _,thresh_m  = cv2.threshold(diff_mono,thresh_v,255,cv2.THRESH_BINARY)
        contours,_  = cv2.findContours(thresh_m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow('Threshold',thresh_m)

        # draw contours
        output = frame.copy()
        cv2.drawContours(output,contours,-1,(0,0,255),2)

        # compute centroid of each contour
        centers = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00']!=0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                centers.append((cx,cy))
        # draw individual centers (optional)
        for (cx,cy) in centers:
            cv2.circle(output,(cx,cy),3,(255,0,0),-1)  # small blue dot

        # draw mean center as green dot
        if centers:
            mx = int(sum(c[0] for c in centers)/len(centers))
            my = int(sum(c[1] for c in centers)/len(centers))
            cv2.circle(output,(mx,my),6,(0,255,0),-1)

    else:
        # blanks until ref set
        cv2.imshow('ColorDiff',np.zeros_like(frame))
        cv2.imshow('RefPreview',np.zeros_like(gray_live))
        cv2.imshow('Threshold',np.zeros_like(gray_live))
        output = frame.copy()

    cv2.imshow('Output',output)

    # --- Key handling ---
    key = cv2.waitKey(1)&0xFF
    if key==ord('r'):
        ref_frame_raw = frame.copy()
        print("Reference frame recorded.")
    elif key==ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
