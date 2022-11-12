import cv2
import numpy as np
from scipy import signal
import time
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore
import pyqtgraph as pg
import matplotlib.pyplot as plt
from signal_processing import Signal_processing
from face_utilities import Face_utilities
import mediapipe as mp


cap = cv2.VideoCapture(0)
pTime = 0

video = False 

# fu = Face_utilities()
sp = Signal_processing()

# right_cheek = [330, 350, 411, 376, 352, 345, 264] 
# left_cheek = [101, 129, 187, 147, 123, 116, 34]
# right_cheek = [348,340] 
cheeks = [329,346,426,100,117,206] # landmark number points 
forehead = [109,107,336,338] 

i=0

t = time.time()
    
#for signal_processing
BUFFER_SIZE = 300
    
fps=0 #for real time capture

    
times = []
data_buffer = []
    
# data for plotting
filtered_data = []
    
fft_of_interest = []
freqs_of_interest = []
    
bpm = 0
    
#plotting
app = QApplication([])
win = pg.GraphicsLayoutWidget()    
#win = pg.GraphicsLayoutWidget(title="plotting")
p1 = win.addPlot(title="detrend")
p2 = win.addPlot(title ="filterd")
win.resize(1200,400)
 # win.show()

p = win.addPlot(title="pure green channel")
# win.resize(600,600)
win.show()

def update():
    p1.clear()
    p1.plot(np.column_stack((freqs_of_interest,fft_of_interest)), pen = 'g')
        
    p2.clear()
    p2.plot(filtered_data[20:],pen='g')             

    p.clear()
    p.plot(data_buffer,pen='g')

    app.processEvents()
            
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(300) # update grapg ทุก ๆ 300ms



mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh # face mesh 
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1) # class นี้รับเเต่ RBG
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # กลับ coordinate เป็น RGB
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks: # ถ้ามีการ detect ได้ 
        for faceLms in results.multi_face_landmarks: # ถ้ามีหลายหน้า ต้องลูปไปทีละหน้า 
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
                                  drawSpec,drawSpec) # วาด mesh ลงไป 
            landmark = []
            for id,lm in enumerate(faceLms.landmark): # convert x y z coordinate เป็น pixel 
                if id in cheeks :
                    #print(lm)
                    ih, iw, ic = img.shape # ดูขนาดเเต่ละ dimension 
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    # print(id,x,y) # id คือ เเต่ละ จุด 1-468
                    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                                  0.7, (0, 255, 0), 1)
                    landmark.append([id,x,y])  
            #### forehead
            # ROIs = img[landmark[1][2]:landmark[0][2],landmark[1][1]:landmark[3][1]]
            # cv2.rectangle(img, (landmark[1][1],landmark[1][2] ), (landmark[3][1],landmark[0][2] ), (255, 0, 255), 1)


            #### cheeks 
            ROIs = (img[landmark[3][2]:landmark[5][2],landmark[3][1]:landmark[4][1]],
                     img[landmark[0][2]:landmark[2][2],landmark[1][1]:landmark[0][1]])
            cv2.rectangle(img, (landmark[3][1], landmark[3][2]), (landmark[4][1], landmark[5][2]), (255, 0, 255), 1) 
            cv2.rectangle(img, (landmark[1][1], landmark[0][2]), (landmark[0][1], landmark[2][2]), (255, 0, 255), 1)  

    green_val = sp.extract_color(ROIs) # ได้ avg ของ green channel
    # print(green_val)
   
    data_buffer.append(green_val)
    
    if(video==False): #  real time
        times.append(time.time() - t) 
    # else:
    #     times.append((1.0/video_fps)*i)
        
    L = len(data_buffer)
    # L = 1,2,3,4,... ไล่ค่า frame เเต่ละเฟรมที่จะไปใส่ใน buffer
    #print("buffer length: " + str(L))
        
    if L > BUFFER_SIZE:
        data_buffer = data_buffer[-BUFFER_SIZE:]
        times = times[-BUFFER_SIZE:]
        #bpms = bpms[-BUFFER_SIZE//2:]
        L = BUFFER_SIZE
    #print(times)
    if L==300: # ถ้ามีเฟรมครบ 100 เฟรม
        fps = float(L) / (times[-1] - times[0])
        # บอก framerate ใน frame 
        cv2.putText(img, "fps: {0:.2f}".format(fps), (30,int(img.shape[0]*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        #
        detrended_data = sp.signal_detrending(data_buffer)
        #print(len(detrended_data))
        #print(len(times))
        filtered_data = sp.butter_bandpass_filter(detrended_data,0.7,4,fps, order = 3)

        interpolated_data = sp.interpolation(filtered_data, times)
            
        normalized_data = sp.normalization(interpolated_data)

        # filtered_data = sp.butter_bandpass_filter(interpolated_data,0.7,4,fps, order = 3)
            
        fft_of_interest, freqs_of_interest = sp.fft(normalized_data, fps)
            
        max_arg = np.argmax(fft_of_interest)
        bpm = freqs_of_interest[max_arg]
        cv2.putText(img, "HR: {}".format(round(bpm)), (int(img.shape[1]*0.8),int(img.shape[0]*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        #print(detrended_data)
        # filtered_data = sp.butter_bandpass_filter(interpolated_data, (bpm-20)/60, (bpm+20)/60, fps, order = 3)
        #print(fps)
        #filtered_data = sp.butter_bandpass_filter(interpolated_data, 0.8, 3, fps, order = 3)
        
        
    # cTime = time.time()
    # fps = 1 / (cTime - pTime)
    # pTime = cTime
    # cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
    #             3, (255, 0, 0), 3)-----

    cv2.imshow("Image", img)
    # cv2.imshow("ROIs_0", ROIs[0])
    # cv2.imshow("ROIs_1", ROIs)
    key = cv2.waitKey(1)
    if key == ord('x'):
         break
    i+=1    
# print(landmark)    

plt.subplot(221)
plt.plot(data_buffer)
plt.subplot(222)
plt.plot(detrended_data)
plt.subplot(223)
# plt.plot(interpolated_data) #세번쨰 파라미터 bandpass filter 영역
# plt.subplot(224)
# plt.xlim(0.0, 7,0) #hz 줄여서
plt.plot(normalized_data)
plt.show()