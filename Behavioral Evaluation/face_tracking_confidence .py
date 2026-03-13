import cv2
import mediapipe as mp
import numpy as np
import time
import math

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# =============================
# Load MediaPipe Face Landmarker
# =============================

face_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

# =============================
# MediaPipe Hand Landmarker (new Tasks API)
# =============================

hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

# Hand landmark index for wrist
WRIST = 0

# Connections for drawing hand skeleton (pairs of landmark indices)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20),# pinky
    (5,9),(9,13),(13,17)           # palm
]

# =============================
# Landmark Indexes
# =============================

RIGHT_EYE = [33, 246, 161, 160, 159, 158, 157,
             173, 133, 155, 154, 153, 145, 144, 163, 7]

LEFT_EYE = [362, 466, 388, 387, 386, 385, 384,
            398, 363, 382, 381, 380, 374, 373, 390, 249]

RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]

L_H_LEFT = [33]
L_H_RIGHT = [133]

R_H_LEFT = [362]
R_H_RIGHT = [263]

# Mouth
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 78
RIGHT_MOUTH = 308

# =============================
# Metric Tracking Variables
# =============================

# --- Blink ---
blink_count = 0
blink_detected = False
EAR_THRESHOLD = 5.5
start_time = time.time()

# --- Lip movement ---
lip_movement_counter = 0
prev_lar = 0
last_lip_active_time = time.time()
lip_stillness_seconds = 0  # total seconds lips were still

# --- Smile ---
smile_frames = 0
total_face_frames = 0
SMILE_LAR_THRESHOLD = 0.15  # lip aspect ratio threshold for smile

# --- Head pose ---
head_stable_frames = 0
head_unstable_frames = 0

# --- Gaze ---
gaze_center_frames = 0
gaze_shift_frames = 0

# --- Hand gesture speed ---
prev_hand_positions = {}       # wrist positions per hand index
prev_hand_time = time.time()
hand_speeds = []               # list of (speed_m_s_approx, timestamp)
# Speed score categories (pixels/sec as proxy, tune as needed)
# Moderate: 50-150 px/s → high confidence
# Fast: >150 px/s → nervous
# Slow/none: <50 px/s → low engagement

# =============================
# Utility Functions
# =============================

def euclidean_distance(p1, p2):
    x1, y1 = p1.ravel()
    x2, y2 = p2.ravel()
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def eye_aspect_ratio(landmarks, right_indices, left_indices):
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclidean_distance(rh_right, rh_left)
    rvDistance = euclidean_distance(rv_top, rv_bottom)
    lhDistance = euclidean_distance(lh_right, lh_left)
    lvDistance = euclidean_distance(lv_top, lv_bottom)

    reRatio = rhDistance / rvDistance if rvDistance != 0 else 0
    leRatio = lhDistance / lvDistance if lvDistance != 0 else 0

    return (reRatio + leRatio) / 2


def iris_position(iris_center, right_point, left_point):
    center_to_right = euclidean_distance(iris_center, right_point)
    total_dist = euclidean_distance(right_point, left_point)
    ratio = center_to_right / total_dist

    if ratio <= 0.42:
        pos = "RIGHT"
    elif ratio <= 0.57:
        pos = "CENTER"
    else:
        pos = "LEFT"

    return pos, ratio


def lip_aspect_ratio(landmarks):
    upper = landmarks[UPPER_LIP]
    lower = landmarks[LOWER_LIP]
    left = landmarks[LEFT_MOUTH]
    right = landmarks[RIGHT_MOUTH]

    vertical = euclidean_distance(upper, lower)
    horizontal = euclidean_distance(left, right)

    if horizontal == 0:
        return 0

    return vertical / horizontal


def score_label(score):
    if score >= 0.9:
        return "HIGH"
    elif score >= 0.6:
        return "MEDIUM"
    else:
        return "LOW"


def compute_confidence_score(
        smile_pct, blink_rate, head_stable_pct,
        lip_active_pct, gaze_center_pct, hand_score):
    """
    Weighted confidence score based on research paper weights:
    Hand Gestures:            30%
    Facial Expressions/Smile: 10%
    Lip Movement:             10%
    Blink Rate:               10%
    Head Movement:            15%
    Gaze Confidence:          10%
    Remaining 15%: combined composite
    """
    # --- Smile score (0-1) ---
    smile_score = min(smile_pct / 60.0, 1.0)  # 60%+ smile rate → max score

    # --- Blink score (0-1): normal <15 bpm → good ---
    if blink_rate <= 12:
        blink_score = 1.0
    elif blink_rate <= 15:
        blink_score = 0.7
    else:
        blink_score = 0.4

    # --- Head stability score ---
    head_score = min(head_stable_pct / 55.0, 1.0)  # 55%+ stable → max

    # --- Lip movement score ---
    lip_score = min(lip_active_pct / 65.0, 1.0)  # 65%+ active → max

    # --- Gaze score ---
    gaze_score = min(gaze_center_pct / 70.0, 1.0)  # 70%+ centered → max

    # --- Hand gesture score ---
    hand_confidence = hand_score  # already 0-1

    # Weighted sum (matching paper weights)
    total = (
        hand_confidence * 0.30 +
        smile_score     * 0.10 +
        lip_score       * 0.10 +
        blink_score     * 0.10 +
        head_score      * 0.15 +
        gaze_score      * 0.10
    )
    # Normalize remaining 25% equally across all factors
    total = total / 0.85  # scale to 0-1 range
    return min(total, 1.0)


# =============================
# Webcam
# =============================

cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = frame.shape
    frame_count += 1

    # Shared mp.Image and timestamp for all Tasks API detectors
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp = int(time.time() * 1000)

    # =============================
    # HAND GESTURE DETECTION (new Tasks API)
    # =============================

    hand_results = hand_landmarker.detect_for_video(mp_image, timestamp)
    current_time = time.time()
    dt = current_time - prev_hand_time

    if hand_results.hand_landmarks:
        for i, hand_lms in enumerate(hand_results.hand_landmarks):
            # Draw hand skeleton
            pts = [
                (int(lm.x * img_w), int(lm.y * img_h))
                for lm in hand_lms
            ]
            for (a, b) in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 200, 255), 1)
            for pt in pts:
                cv2.circle(frame, pt, 3, (0, 255, 200), -1)

            # Track wrist speed
            wx, wy = pts[WRIST]
            if i in prev_hand_positions and dt > 0:
                px, py = prev_hand_positions[i]
                pixel_dist = math.sqrt((wx - px) ** 2 + (wy - py) ** 2)
                speed_px_per_sec = pixel_dist / dt
                hand_speeds.append((speed_px_per_sec, current_time))

            prev_hand_positions[i] = (wx, wy)

    prev_hand_time = current_time

    # Compute recent hand speed (last 5 seconds)
    hand_speeds = [(s, t) for s, t in hand_speeds if current_time - t < 5]
    if hand_speeds:
        avg_speed = np.mean([s for s, _ in hand_speeds])
        # Score: moderate 50-150 → high confidence
        if 50 <= avg_speed <= 150:
            current_hand_score = 1.0
            hand_label = "MODERATE (confident)"
        elif avg_speed > 150:
            current_hand_score = 0.5
            hand_label = "FAST (nervous)"
        else:
            current_hand_score = 0.6
            hand_label = "SLOW/STILL"
    else:
        avg_speed = 0
        current_hand_score = 0.5
        hand_label = "NO HANDS"

    # =============================
    # FACE TRACKING
    # =============================

    results = face_landmarker.detect_for_video(mp_image, timestamp)

    elapsed_time = time.time() - start_time

    if results.face_landmarks:
        total_face_frames += 1
        landmarks = results.face_landmarks[0]

        mesh_points = np.array([
            np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
            for p in landmarks
        ])

        # =============================
        # IRIS / GAZE TRACKING
        # =============================

        (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
        (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

        center_left  = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)

        cv2.circle(frame, center_left, int(l_radius), (255, 0, 255), 1)
        cv2.circle(frame, center_right, int(r_radius), (255, 0, 255), 1)

        right_iris_pos, right_eye_ratio = iris_position(
            center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT[0]])
        left_iris_pos, left_eye_ratio = iris_position(
            center_left, mesh_points[L_H_RIGHT], mesh_points[L_H_LEFT[0]])

        avg_ratio = (right_eye_ratio + left_eye_ratio) / 2

        if avg_ratio <= 0.42:
            eye_pos = "RIGHT"
        elif avg_ratio <= 0.57:
            eye_pos = "CENTER"
            gaze_center_frames += 1
        else:
            eye_pos = "LEFT"

        if eye_pos != "CENTER":
            gaze_shift_frames += 1

        # =============================
        # BLINK DETECTION
        # =============================

        ear = eye_aspect_ratio(mesh_points, RIGHT_EYE, LEFT_EYE)

        if ear > EAR_THRESHOLD:
            blink_detected = True
        else:
            if blink_detected:
                blink_count += 1
                blink_detected = False

        blink_rate = (blink_count / elapsed_time) * 60 if elapsed_time > 0 else 0

        # =============================
        # LIP / SMILE TRACKING
        # =============================

        lar = lip_aspect_ratio(mesh_points)

        # Smile detection
        if lar > SMILE_LAR_THRESHOLD:
            smile_frames += 1

        # Lip movement
        if abs(lar - prev_lar) > 0.01:
            lip_movement_counter += 1
            last_lip_active_time = time.time()
        else:
            lip_stillness_seconds += (1 / 30.0)  # approx per frame at 30fps

        prev_lar = lar

        # =============================
        # HEAD POSE ESTIMATION
        # =============================

        face_2d = []
        face_3d = []

        for idx, lm in enumerate(landmarks):
            if idx in [33, 263, 1, 61, 291, 199]:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])
                if idx == 1:
                    nose_2d = (x, y)
                    nose_3d = (x, y, lm.z * 3000)

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = img_w
        cam_matrix = np.array([
            [focal_length, 0, img_h / 2],
            [0, focal_length, img_w / 2],
            [0, 0, 1]
        ])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success_pnp, rot_vec, trans_vec = cv2.solvePnP(
            face_3d, face_2d, cam_matrix, dist_matrix)

        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        x_ang = angles[0] * 360
        y_ang = angles[1] * 360
        z_ang = angles[2] * 360

        head_stable = abs(x_ang) <= 10 and abs(y_ang) <= 10
        if head_stable:
            head_stable_frames += 1
            head_direction = "Forward"
        else:
            head_unstable_frames += 1
            if y_ang < -10:
                head_direction = "Left"
            elif y_ang > 10:
                head_direction = "Right"
            elif x_ang < -10:
                head_direction = "Down"
            else:
                head_direction = "Up"

        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_2d[0] + y_ang * 10), int(nose_2d[1] - x_ang * 10))
        cv2.line(frame, p1, p2, (255, 0, 0), 3)

        # =============================
        # COMPUTE LIVE SCORES
        # =============================

        smile_pct   = (smile_frames / total_face_frames * 100) if total_face_frames > 0 else 0
        head_stable_pct = (head_stable_frames / total_face_frames * 100) if total_face_frames > 0 else 0
        gaze_center_pct = (gaze_center_frames / total_face_frames * 100) if total_face_frames > 0 else 0
        lip_active_pct  = (lip_movement_counter / max(total_face_frames, 1)) * 100

        overall_score = compute_confidence_score(
            smile_pct, blink_rate, head_stable_pct,
            lip_active_pct, gaze_center_pct, current_hand_score
        )

        # =============================
        # DISPLAY LIVE METRICS
        # =============================

        # Left panel - metrics
        y_offset = 30
        line_h = 35

        def put_metric(label, value_str, score_str, color, y):
            cv2.putText(frame, f"{label}: {value_str} [{score_str}]",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        # 1. Smile
        smile_color = (0, 255, 0) if smile_pct >= 60 else (0, 165, 255) if smile_pct >= 30 else (0, 0, 255)
        put_metric("Smile", f"{smile_pct:.1f}%", score_label(smile_pct / 100), smile_color, y_offset)
        y_offset += line_h

        # 2. Lip Movement
        lip_color = (0, 255, 255)
        put_metric("Lip Moves", f"{lip_movement_counter}", f"LAR:{lar:.3f}", lip_color, y_offset)
        y_offset += line_h

        # 3. Blink Rate
        blink_color = (0, 255, 0) if blink_rate <= 12 else (0, 165, 255) if blink_rate <= 15 else (0, 0, 255)
        put_metric("Blink Rate", f"{blink_rate:.1f} bpm", score_label(1.0 if blink_rate<=12 else 0.7 if blink_rate<=15 else 0.4), blink_color, y_offset)
        y_offset += line_h

        # 4. Head Pose
        head_color = (0, 255, 0) if head_stable else (0, 0, 255)
        put_metric("Head", f"{head_direction}", f"Stable:{head_stable_pct:.0f}%", head_color, y_offset)
        y_offset += line_h

        # 5. Gaze
        gaze_color = (0, 255, 0) if eye_pos == "CENTER" else (0, 165, 255)
        put_metric("Gaze", f"{eye_pos}", f"Center:{gaze_center_pct:.0f}%", gaze_color, y_offset)
        y_offset += line_h

        # 6. Hand Gesture
        hand_color = (0, 255, 0) if current_hand_score >= 0.9 else (0, 165, 255) if current_hand_score >= 0.6 else (0, 0, 255)
        put_metric("Hands", f"{avg_speed:.0f}px/s", hand_label[:10], hand_color, y_offset)
        y_offset += line_h

        # Overall confidence score
        conf_color = (0, 255, 0) if overall_score >= 0.7 else (0, 165, 255) if overall_score >= 0.5 else (0, 0, 255)
        cv2.putText(frame, f"CONFIDENCE: {overall_score:.2f} [{score_label(overall_score)}]",
                    (10, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, conf_color, 2)

        # EAR display
        cv2.putText(frame, f"EAR: {ear:.2f}",
                    (img_w - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Head pose angles
        cv2.putText(frame, f"X:{x_ang:.1f} Y:{y_ang:.1f}",
                    (img_w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)

        # Session timer
        cv2.putText(frame, f"Time: {int(elapsed_time)}s",
                    (img_w - 130, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    else:
        cv2.putText(frame, "No face detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, "Press Q to quit & see summary",
                (10, img_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("Confidence Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hand_landmarker.close()

# =============================
# FINAL SUMMARY REPORT
# =============================

elapsed_time = time.time() - start_time
blink_rate_final = (blink_count / elapsed_time) * 60 if elapsed_time > 0 else 0
smile_pct_final       = (smile_frames / total_face_frames * 100)    if total_face_frames > 0 else 0
head_stable_pct_final = (head_stable_frames / total_face_frames * 100) if total_face_frames > 0 else 0
gaze_center_pct_final = (gaze_center_frames / total_face_frames * 100) if total_face_frames > 0 else 0
lip_active_pct_final  = (lip_movement_counter / max(total_face_frames, 1)) * 100

avg_hand_speed_final = np.mean([s for s, _ in hand_speeds]) if hand_speeds else 0
if 50 <= avg_hand_speed_final <= 150:
    hand_label_final = "MODERATE (confident)"
    hand_score_final = 1.0
elif avg_hand_speed_final > 150:
    hand_label_final = "FAST (nervous)"
    hand_score_final = 0.5
else:
    hand_label_final = "SLOW/STILL"
    hand_score_final = 0.6

final_score = compute_confidence_score(
    smile_pct_final, blink_rate_final, head_stable_pct_final,
    lip_active_pct_final, gaze_center_pct_final, hand_score_final
)

def interpret_blink(rate):
    if rate <= 12:   return "Normal – HIGH confidence"
    elif rate <= 15: return "Slightly elevated – MEDIUM confidence"
    else:            return "Excessive – LOW confidence (stress/overload)"

def interpret_smile(pct):
    if pct >= 60:   return "Frequent – HIGH confidence"
    elif pct >= 25: return "Occasional – MEDIUM confidence"
    else:           return "Rare – LOW confidence"

def interpret_head(pct):
    if pct >= 55:   return "Stable – HIGH confidence"
    elif pct >= 30: return "Moderate deviation – MEDIUM confidence"
    else:           return "Frequent movement – LOW confidence"

def interpret_gaze(pct):
    if pct >= 70:   return "Steady – HIGH confidence"
    elif pct >= 50: return "Occasional shifts – MEDIUM confidence"
    else:           return "Frequent shifts – LOW confidence"

def interpret_lip(pct):
    if pct >= 65:   return "Active – HIGH confidence"
    elif pct >= 30: return "Occasional pauses – MEDIUM confidence"
    else:           return "Prolonged stillness – LOW confidence"

print("\n" + "=" * 60)
print("         CONFIDENCE DETECTION — FINAL SUMMARY REPORT")
print("=" * 60)
print(f"  Session Duration      : {int(elapsed_time)} seconds")
print(f"  Total Frames (face)   : {total_face_frames}")
print("-" * 60)
print(f"  1. SMILE")
print(f"     Smiling frames     : {smile_frames} ({smile_pct_final:.1f}% of session)")
print(f"     Interpretation     : {interpret_smile(smile_pct_final)}")
print(f"     Paper weight       : 10%")
print()
print(f"  2. LIP MOVEMENT")
print(f"     Movement events    : {lip_movement_counter}")
print(f"     Stillness time     : ~{int(lip_stillness_seconds)}s")
print(f"     Activity rate      : {lip_active_pct_final:.1f}%")
print(f"     Interpretation     : {interpret_lip(lip_active_pct_final)}")
print(f"     Paper weight       : 10%")
print()
print(f"  3. BLINK RATE")
print(f"     Total blinks       : {blink_count}")
print(f"     Blinks per minute  : {blink_rate_final:.1f} bpm")
print(f"     Interpretation     : {interpret_blink(blink_rate_final)}")
print(f"     Paper weight       : 10%")
print()
print(f"  4. HEAD POSE STABILITY")
print(f"     Stable frames      : {head_stable_frames} ({head_stable_pct_final:.1f}%)")
print(f"     Unstable frames    : {head_unstable_frames} ({100-head_stable_pct_final:.1f}%)")
print(f"     Interpretation     : {interpret_head(head_stable_pct_final)}")
print(f"     Paper weight       : 15%")
print()
print(f"  5. GAZE DIRECTION")
print(f"     Center gaze frames : {gaze_center_frames} ({gaze_center_pct_final:.1f}%)")
print(f"     Gaze shifts        : {gaze_shift_frames}")
print(f"     Interpretation     : {interpret_gaze(gaze_center_pct_final)}")
print(f"     Paper weight       : 10%")
print()
print(f"  6. HAND GESTURES")
print(f"     Avg. wrist speed   : {avg_hand_speed_final:.1f} px/s")
print(f"     Classification     : {hand_label_final}")
print(f"     Confidence score   : {hand_score_final:.2f}")
print(f"     Paper weight       : 30%")
print("-" * 60)
print(f"  OVERALL CONFIDENCE SCORE : {final_score:.3f}  [{score_label(final_score)}]")
print("=" * 60)
