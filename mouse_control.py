import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import threading

pyautogui.FAILSAFE = False 


capture_hands = mp.solutions.hands.Hands(max_num_hands=2)
drawing_option = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 

sensitivity = 1.5  
smoothing = 10  
cursor_speed = 4.0  

prev_x, prev_y = 0, 0

def map_coordinates(x, y, sensitivity):
    mapped_x = np.interp(x, (0, 1), (0, screen_width)) * sensitivity
    mapped_y = np.interp(y, (0, 1), (0, screen_height)) * sensitivity
    return mapped_x, mapped_y

def process_frame(frame):
    global prev_x, prev_y, sensitivity, cursor_speed
    
  
    frame = cv2.flip(frame, 1)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output_hand = capture_hands.process(rgb_image)
    all_hands = output_hand.multi_hand_landmarks
    
    if all_hands:
        for hand in all_hands:
            drawing_option.draw_landmarks(frame, hand)
            one_hand_landmarks = hand.landmark
            
            for id, lm in enumerate(one_hand_landmarks):
                x, y = lm.x, lm.y
                if id == 8:  
                    mouse_x, mouse_y = map_coordinates(x, y, sensitivity)
                    
              
                    mouse_x = prev_x + (mouse_x - prev_x) / smoothing
                    mouse_y = prev_y + (mouse_y - prev_y) / smoothing
                    
                  
                    mouse_x = prev_x + (mouse_x - prev_x) * cursor_speed
                    mouse_y = prev_y + (mouse_y - prev_y) * cursor_speed
                    
                  
                    pyautogui.moveTo(mouse_x, mouse_y)
                    
                 
                    prev_x, prev_y = mouse_x, mouse_y
                  
                    cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 15, (255, 0, 255), cv2.FILLED)
                
                if id == 4:  
                    x2, y2 = int(x * frame.shape[1]), int(y * frame.shape[0])
                    cv2.circle(frame, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            
           
            thumb_tip = hand.landmark[4]
            index_tip = hand.landmark[8]
            dist = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
            
      
            if dist < 0.01:
                pyautogui.click()
                print("click")
    
    return frame

def main_loop():
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        processed_frame = process_frame(frame)     
    
        cv2.imshow("Hand movement video capture", processed_frame)
        

        key = cv2.waitKey(1)
        if key == 27:  # Esc
            break
        elif key == ord('+'):  
            global sensitivity
            sensitivity += 0.1
            print(f"Sensitivity: {sensitivity}")
        elif key == ord('-'):  
            sensitivity = max(0.1, sensitivity - 0.1)
            print(f"Sensitivity: {sensitivity}")
        elif key == ord('>'):  
            global cursor_speed
            cursor_speed += 0.1
            print(f"Cursor Speed: {cursor_speed}")
        elif key == ord('<'):  
            cursor_speed = max(0.1, cursor_speed - 0.1)
            print(f"Cursor Speed: {cursor_speed}")

thread = threading.Thread(target=main_loop)
thread.start()
thread.join()

camera.release()
cv2.destroyAllWindows()