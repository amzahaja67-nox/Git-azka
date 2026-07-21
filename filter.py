import cv2
import mediapipe as mp
import numpy as np

def apply_filter(roi, filter_name):
    if roi.size == 0:
        return roi
    if filter_name == "normal":
        return roi
    elif filter_name == "gray":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif filter_name == "sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        return cv2.transform(roi, kernel)
    elif filter_name == "invert":
        return cv2.bitwise_not(roi)
    elif filter_name == "thermal":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    elif filter_name == "dotted":
        h, w = roi.shape[:2]
        small = cv2.resize(roi, (w//10, h//10), interpolation=cv2.INTER_LINEAR)
        dotted = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return dotted
    elif filter_name == "pop_art":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        return cv2.applyColorMap(thresh, cv2.COLORMAP_HOT)
    return roi

FILTROS = ["normal", "dotted", "gray", "thermal", "sepia", "pop_art", "invert"]
filtro_index = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def get_fingertips(hand_landmarks, w, h):
    thumb = hand_landmarks.landmark[4]
    index = hand_landmarks.landmark[8]
    return (int(thumb.x * w), int(thumb.y * h)), (int(index.x * w), int(index.y * h))

def render_portal(frame, p1, p2, p3, p4, filter_name):
    h, w = frame.shape[:2]
    pts = np.array([p1, p2, p4, p3], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    rect = cv2.minAreaRect(pts)
    box_w, box_h = int(rect[1][0]), int(rect[1][1])
    if box_w < 10 or box_h < 10:
        return frame

    src_pts = np.float32([p1, p2, p3, p4])
    dst_pts = np.float32([[0,0], [box_w,0], [0,box_h], [box_w,box_h]])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, M, (box_w, box_h))
    warped_filtered = apply_filter(warped, filter_name)

    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    unwarped = cv2.warpPerspective(warped_filtered, M_inv, (w, h))

    result = frame.copy()
    result[mask == 255] = unwarped[mask == 255]

    cv2.polylines(result, [pts], True, (255,255,255), 1)
    for i in range(1, 5):
        cv2.line(result,
                 (p1[0] + (p3[0]-p1[0])*i//5, p1[1] + (p3[1]-p1[1])*i//5),
                 (p2[0] + (p4[0]-p2[0])*i//5, p2[1] + (p4[1]-p2[1])*i//5),
                 (255,255,255), 1, cv2.LINE_AA)

    return result

def main():
    global filtro_index
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    last_switch = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        points = []

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            for hand_lms in results.multi_hand_landmarks:
                thumb, index = get_fingertips(hand_lms, w, h)
                points.append(thumb)
                points.append(index)
                cv2.circle(frame, thumb, 8, (0,255,0), -1)
                cv2.circle(frame, index, 8, (0,255,0), -1)

            if len(points) == 4:
                points = sorted(points, key=lambda p: p[0])
                left = sorted(points[:2], key=lambda p: p[1])
                right = sorted(points[2:], key=lambda p: p[1])
                p1, p2, p3, p4 = left[0], left[1], right[0], right[1]

                frame = render_portal(frame, p1, p2, p3, p4, FILTROS[filtro_index])

                dist = np.linalg.norm(np.array(p1) - np.array(p3))
                if dist < 100 and (cv2.getTickCount() - last_switch)/cv2.getTickFrequency() > 1.5:
                    filtro_index = (filtro_index + 1) % len(FILTROS)
                    last_switch = cv2.getTickCount()

        cv2.imshow("Hand Portal - projek coding filter", frame)
        if cv2.waitKey(1) & 0xFF == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()