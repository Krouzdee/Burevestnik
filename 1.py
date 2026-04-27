from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

model = YOLO("last.pt")

font = ImageFont.truetype("arial.ttf", 20)


def draw_text_cv2(
    img, text, position, font, text_color=(255, 255, 255), bg_color=(255, 0, 0)
):

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    x, y = position

    bbox = draw.textbbox((x, y), text, font=font)
    x1, y1, x2, y2 = bbox

    pad = 4

    draw.rectangle((x1 - pad, y1 - pad, x2 + pad, y2 + pad), fill=bg_color)

    draw.text((x, y), text, font=font, fill=text_color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


video_path = "cam0.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Ошибка открытия {video_path}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("out.mp4", fourcc, fps, (width, height))

track_history = defaultdict(lambda: [])
track_last_seen = {}

MAX_MISSED = int(fps * 1)
frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Конец видео")
        break

    frame_idx += 1

    results = model.track(frame, persist=True, conf=0.5)

    annotated_frame = frame.copy()

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        names = model.names

        for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = map(int, box)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            track = track_history[track_id]
            track.append((cx, cy))
            if len(track) > 30:
                track.pop(0)

            track_last_seen[track_id] = frame_idx

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"ID:{track_id} {names[cls_id]} ({cx},{cy})"
            text_y = y1 - 25 if y1 - 25 > 10 else y1 + 25

            annotated_frame = draw_text_cv2(
                annotated_frame,
                label,
                (x1, text_y),
                font,
                text_color=(255, 255, 255),
                bg_color=(0, 0, 255),
            )

    for track_id, track in track_history.items():
        if frame_idx - track_last_seen.get(track_id, 0) > MAX_MISSED:
            continue

        if len(track) < 2:
            continue

        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

        cv2.polylines(
            annotated_frame,
            [points],
            isClosed=False,
            color=(230, 230, 230),
            thickness=2,
        )

    to_delete = [
        tid
        for tid in track_history
        if frame_idx - track_last_seen.get(tid, 0) > MAX_MISSED
    ]

    for tid in to_delete:
        track_history.pop(tid, None)
        track_last_seen.pop(tid, None)

    cv2.imshow("YOLO Tracking", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
