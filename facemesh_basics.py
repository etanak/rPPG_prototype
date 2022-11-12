import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

right_cheek = [330, 350, 411, 376, 352, 345, 264] 
left_cheek = [101, 129, 187, 147, 123, 116, 34]
forehead = [109,107,336,338] 

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh # face mesh 
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2) # class นี้รับเเต่ RBG
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
                if id in forehead :
                    #print(lm)
                    ih, iw, ic = img.shape # ดูขนาดเเต่ละ dimension 
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    # print(id,x,y) # id คือ เเต่ละ จุด 1-468
                    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                                  0.7, (0, 255, 0), 1)
                    landmark.append([id,x,y]) 
            ROIs = img[landmark[1][2]:landmark[0][2],landmark[1][1]:landmark[3][1]]
            cv2.rectangle(img, (landmark[1][1],landmark[1][2] ), (landmark[3][1],landmark[0][2] ), (255, 0, 255), 1)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    cv2.imshow("ROIs",ROIs)            
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('x'):
         break