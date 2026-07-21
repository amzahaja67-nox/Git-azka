import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)
cap = cv2.VideoCapture(0)

def is_peace(hand_landmarks):
    
    def finger_up(tip, pip):
        return hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y

    index_up = finger_up(8, 6)
    middle_up = finger_up(12, 10)
    ring_down = not finger_up(16, 14)
    pinky_down = not finger_up(20, 18)

    return index_up and middle_up and ring_down and pinky_down

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    output = frame.copy()
    peace = False

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            if is_peace(hand):
                peace = True

    if peace:
        
        output = cv2.GaussianBlur(output, (51, 51), 0)
        cv2.putText(output, "BLUUUUUUUUUUUURRRR", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(output, "Foto kita..... ", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("ASTAGA NAGA - Blur Filter", output)

    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
