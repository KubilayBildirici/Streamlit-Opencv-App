import cv2
import mediapipe as mp
import time
import streamlit as st


def Bodydetect():
    frame_placeholder = st.empty()
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        #imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)
        print(results.pose_landmarks)

        if results.pose_landmarks:
            mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
        frame_placeholder.image(frame, channels="RGB")

        cv2.imshow("img", frame)
        cv2.waitKey(10)


def Handmesh():
    ### LOAD A VIDEO
    cap = cv2.VideoCapture(0)

    frame_placeholder = st.empty()


    ## call hand mesh funnction
    mpHand = mp.solutions.hands
    hand = mpHand.Hands()
    mpDraw = mp.solutions.drawing_utils

    while True:
        success, frame = cap.read()
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, handLms, mpHand.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

        frame_placeholder.image(frame, channels="RGB")



        cv2.imshow("img", frame)
        cv2.waitKey(10)


def Facedetect():
    ### LOAD A VIDEO
    cap = cv2.VideoCapture(0)

    frame_placeholder = st.empty()

    mpFace = mp.solutions.face_mesh
    facemesh = mpFace.FaceMesh(max_num_faces=2)
    mpdraw = mp.solutions.drawing_utils

    draw_spec = mpdraw.DrawingSpec(thickness=1,circle_radius=1,color = (255,0,255))
    while True:
        success, frame = cap.read()
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = facemesh.process(imgRGB)
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mpdraw.draw_landmarks(frame,faceLms,mpFace.FACEMESH_TESSELATION,draw_spec,draw_spec)
            for id,lm in enumerate(faceLms.landmark):
                h,w,_ = frame.shape
                cx,cy = int(lm.x*w),int(lm.y*h)

        frame_placeholder.image(frame, channels="RGB")

        cv2.imshow("img", frame)
        cv2.waitKey(10)
















