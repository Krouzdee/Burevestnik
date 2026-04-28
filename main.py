import customtkinter as ctk
from CTkTable import CTkTable
import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from pygrabber.dshow_graph import FilterGraph
from collections import defaultdict
import numpy as np
import torch
from tkinter import filedialog
import winsound

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

model = YOLO("best_drone_detect_scale.pt")
model.to(DEVICE)

names = {0: "Самолёт", 1: "Квадракоптер", 2: "Вертолёт"}
colors = {
    0: (0, 0, 128),
    1: (0, 128, 0),
    2: (128, 0, 0),
}

font = ImageFont.truetype("arial.ttf", 40)
font_table = ImageFont.truetype("arial.ttf", 16)


def center(work, x: int, y: int) -> str:
    POS_X = work.winfo_screenwidth() // 2 - x // 2
    POS_Y = work.winfo_screenheight() // 2 - y // 2
    return f"{x}x{y}+{POS_X}+{POS_Y}"


def bgr_to_rgb(c):
    return (c[2], c[1], c[0])


def draw_text_cv2(
    img, text, position, font, text_color=(255, 255, 255), bg_color=(255, 0, 0)
):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    x, y = position
    bbox = draw.textbbox((x, y), text, font=font)
    x1, y1, x2, y2 = bbox
    pad = 8
    draw.rectangle((x1 - pad, y1 - pad, x2 + pad, y2 + pad), fill=bg_color)
    draw.text((x, y), text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class BurevestnikApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("БУРЕВЕСТНИК - Система мониторинга БПЛА")
        self.geometry(center(self, 1280, 720))
        self.resizable(width=False, height=False)

        self.cap = None
        self.video_source_type = None
        self.drawing_roi = False
        self.roi_drawing_active = False
        self.roi_start_x = 0
        self.roi_start_y = 0
        self.roi_coords = [100, 100, 300, 300]
        self.roi_alert_active = False
        self.last_table_data = None
        self.is_hide = False

        self.track_history = defaultdict(lambda: [])
        self.track_last_seen = {}
        self.track_class = {}
        self.fps = 30
        self.frame_idx = 0
        self.current_detections = []

        self.current_frame_shape = (0, 0)

        self.video_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="#000000")
        self.video_frame.place(relx=0.01, rely=0.03, relwidth=0.48, relheight=0.94)

        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

        self.video_label.bind("<ButtonPress-1>", self.on_mouse_down)
        self.video_label.bind("<B1-Motion>", self.on_mouse_move)
        self.video_label.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.control_panel = ctk.CTkFrame(self, corner_radius=15)
        self.control_panel.place(relx=0.51, rely=0.03, relwidth=0.48, relheight=0.94)

        self.title_label = ctk.CTkLabel(
            self.control_panel,
            text="УПРАВЛЕНИЕ СИСТЕМОЙ",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.title_label.place(relx=0.5, y=30, anchor="center")

        self.source_label = ctk.CTkLabel(
            self.control_panel,
            text="Источник видео:",
            font=ctk.CTkFont(size=14),
        )
        self.source_label.place(x=30, y=70)

        self.source_type_var = ctk.StringVar(value="camera")
        self.radio_camera = ctk.CTkRadioButton(
            self.control_panel,
            text="Камера",
            variable=self.source_type_var,
            value="camera",
            command=self.on_source_type_change,
        )
        self.radio_camera.place(x=160, y=70)

        self.radio_file = ctk.CTkRadioButton(
            self.control_panel,
            text="Видеофайл",
            variable=self.source_type_var,
            value="file",
            command=self.on_source_type_change,
        )
        self.radio_file.place(x=280, y=70)

        self.cam_select = ctk.CTkOptionMenu(
            self.control_panel,
            values=self.get_available_cameras(),
            command=self.change_camera,
        )
        self.cam_select.place(x=30, y=110, relwidth=0.7)

        self.btn_browse = ctk.CTkButton(
            self.control_panel,
            text="Обзор",
            width=80,
            height=30,
            command=self.browse_video_file,
        )
        self.btn_browse.place(x=30, y=110, relwidth=0.7)
        self.btn_browse.place_forget()

        self.file_path_label = ctk.CTkLabel(
            self.control_panel,
            text="",
            font=ctk.CTkFont(size=10),
            text_color="#888888",
        )
        self.file_path_label.place(x=30, y=145)

        self.btn_roi = ctk.CTkButton(
            self.control_panel,
            text="Включить зону интереса",
            height=40,
            fg_color="#2b7b50",
            command=self.toggle_roi_mode,
        )
        self.btn_roi.place(x=30, y=180, relwidth=0.9)

        self.table_frame = ctk.CTkFrame(self.control_panel, fg_color="transparent")
        self.table_frame.place(x=15, y=240, relwidth=0.95, relheight=0.45)

        self.table = CTkTable(
            master=self.table_frame,
            row=6,
            column=5,
            values=[["ID", "Тип", "X", "Y", "Статус"]] + [["", "", "", "", ""]] * 5,
            header_color="#1a1a2e",
            hover_color="#16213e",
            text_color="#e0e0e0",
        )
        self.table.pack(expand=True, fill="both", padx=5, pady=5)

        self.status_label = ctk.CTkLabel(
            self.control_panel,
            text="СТАТУС: ГОТОВ",
            font=("Arial", 13, "bold"),
            text_color="#aaaaaa",
        )
        self.status_label.place(relx=0.5, rely=0.95, anchor="center")

        self.hide_switch = ctk.CTkSwitch(
            self.control_panel,
            text="Отображать трекеры",
            height=10,
            width=50,
            command=self.change_hide,
        )
        self.hide_switch.place(x=30, y=150)

        self.hide_switch.select()

        self.video_file_path = None

        if self.cam_select.get() != "Камеры не найдены":
            self.change_camera(self.cam_select.get())

        self.update_frame()

    def change_hide(self):
        self.is_hide = not self.is_hide

    def tables_are_equal(self, data1, data2):
        if data1 is None or data2 is None:
            return False
        if len(data1) != len(data2):
            return False
        for row1, row2 in zip(data1, data2):
            if row1 != row2:
                return False
        return True

    def on_source_type_change(self):
        if self.source_type_var.get() == "camera":
            self.btn_browse.place_forget()
            self.cam_select.place(x=30, y=110, relwidth=0.7)
            self.file_path_label.configure(text="")
            if (
                self.cam_select.get() != "Камеры не найдены"
                and self.video_source_type == "file"
            ):
                self.change_camera(self.cam_select.get())
        else:
            self.cam_select.place_forget()
            self.btn_browse.place(x=30, y=110, relwidth=0.7)

    def browse_video_file(self):
        file_path = filedialog.askopenfilename(
            title="Выберите видеофайл",
            filetypes=[("Видеофайлы", "*.mp4 *.avi *.mov *.mkv"), ("Все файлы", "*.*")],
        )
        if file_path:
            self.video_file_path = file_path
            short_name = file_path.split("/")[-1]
            if len(short_name) > 40:
                short_name = short_name[:37] + "..."
            self.file_path_label.configure(text=short_name)
            self.open_video_file(file_path)

    def open_video_file(self, file_path):
        if self.cap:
            self.cap.release()
            self.track_history.clear()
            self.track_last_seen.clear()
            self.track_class.clear()
            self.frame_idx = 0
        self.cap = cv2.VideoCapture(file_path)
        self.video_source_type = "file"
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30

    def _event_to_frame_coords(self, event):
        cam_h, cam_w = self.current_frame_shape[:2]
        if cam_w == 0:
            return None, None

        display_w = self.video_label.winfo_width()
        display_h = self.video_label.winfo_height()

        if display_w < 10:
            return None, None

        scale_x = cam_w / display_w
        scale_y = cam_h / display_h

        return event.x * scale_x, event.y * scale_y

    def on_mouse_down(self, event):
        if not self.drawing_roi or self.cap is None:
            return
        fx, fy = self._event_to_frame_coords(event)
        if fx is None:
            return
        self.roi_drawing_active = True
        self.roi_start_x, self.roi_start_y = fx, fy

    def on_mouse_move(self, event):
        if not self.drawing_roi or not self.roi_drawing_active:
            return
        fx, fy = self._event_to_frame_coords(event)
        if fx is None:
            return

        cam_h, cam_w = self.current_frame_shape[:2]

        x = max(0, min(self.roi_start_x, fx))
        y = max(0, min(self.roi_start_y, fy))
        w = abs(fx - self.roi_start_x)
        h = abs(fy - self.roi_start_y)

        w = min(w, cam_w - x)
        h = min(h, cam_h - y)

        self.roi_coords = [x, y, w, h]

    def on_mouse_up(self, event):
        self.roi_drawing_active = False

    def toggle_roi_mode(self):
        self.drawing_roi = not self.drawing_roi
        self.roi_drawing_active = False
        color = "#8b2e2e" if self.drawing_roi else "#2b7b50"
        txt = (
            "Выключить зону интереса" if self.drawing_roi else "Включить зону интереса"
        )
        self.btn_roi.configure(text=txt, fg_color=color)

    def get_available_cameras(self):
        devices = FilterGraph().get_input_devices()

        if not devices:
            return ["Камеры не найдены"]

        return devices

    def change_camera(self, choice):
        if self.cap:
            self.cap.release()
            self.track_history.clear()
            self.track_last_seen.clear()
            self.track_class.clear()
            self.frame_idx = 0
        try:
            idx = self.cam_select._values.index(choice)
            self.cap = cv2.VideoCapture(idx)
            self.video_source_type = "camera"
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30
        except:
            pass

    def process_frame(self, frame):
        self.frame_idx += 1
        self.current_detections = []

        original_h, original_w = frame.shape[:2]
        MAX_SIZE = 640

        scale = min(MAX_SIZE / original_w, MAX_SIZE / original_h)

        if scale < 1.0:
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)
            resized_frame = cv2.resize(
                frame, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
        else:
            resized_frame = frame
            scale = 1.0

        results = model.track(
            resized_frame,
            persist=True,
            conf=0.7,
            verbose=False,
            imgsz=640,
            device=DEVICE,
            half=True if DEVICE == "cuda" else False,
        )

        max_missed = int(self.fps * 1)
        active_ids = set()

        track_layer = frame.copy()

        beep = False
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes_xyxy = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls_id in zip(boxes_xyxy, track_ids, class_ids):
                active_ids.add(track_id)

                x1, y1, x2, y2 = box.tolist()
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                track = self.track_history[track_id]
                track.append((cx, cy))
                if len(track) > 200:
                    track.pop(0)

                self.track_last_seen[track_id] = self.frame_idx
                self.track_class[track_id] = cls_id

                status = "НОРМА"
                if (
                    self.drawing_roi
                    and self.box_intersects_roi(x1, y1, x2, y2)
                    and not beep
                ):
                    if self.frame_idx % 5 == 0:
                        self.after(1, lambda: winsound.Beep(1000, 500))
                        beep = True
                    status = "ТРЕВОГА"

                self.current_detections.append((track_id, cls_id, cx, cy, status))

                cv2.rectangle(
                    track_layer,
                    (x1, y1),
                    (x2, y2),
                    colors.get(cls_id, (255, 255, 255)),
                    3,
                )

                label = f"ID:{track_id} {names.get(cls_id, 'Объект')} ({cx},{cy})"
                text_y = y1 - 40 if y1 - 40 > 10 else y1 + 40

                track_layer = draw_text_cv2(
                    track_layer,
                    label,
                    (x1, text_y),
                    font,
                    text_color=(255, 255, 255),
                    bg_color=bgr_to_rgb(colors.get(cls_id, (128, 128, 128))),
                )

            if not self.is_hide:
                for track_id, track in self.track_history.items():
                    if track_id not in active_ids:
                        continue
                    if (
                        self.frame_idx - self.track_last_seen.get(track_id, 0)
                        > max_missed
                    ):
                        continue
                    if len(track) < 2:
                        continue

                    cls_id = self.track_class.get(track_id, 0)
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

                    cv2.polylines(
                        track_layer,
                        [points],
                        isClosed=False,
                        color=colors.get(cls_id, (0, 255, 255)),
                        thickness=10,
                    )

        frame = track_layer

        to_delete = [
            tid
            for tid in self.track_history
            if self.frame_idx - self.track_last_seen.get(tid, 0) > max_missed
        ]

        for tid in to_delete:
            self.track_history.pop(tid, None)
            self.track_last_seen.pop(tid, None)
            self.track_class.pop(tid, None)

        return frame

    def update_table(self):
        detections_sorted = sorted(self.current_detections, key=lambda d: d[0])[:5]

        table_data = [["ID", "Тип", "X", "Y", "Статус"]]
        intrusion_detected = False

        for i in range(5):
            if i < len(detections_sorted):
                track_id, cls_id, cx, cy, status = detections_sorted[i]
                obj_type = names.get(cls_id, "Объект")
                table_data.append([str(track_id), obj_type, str(cx), str(cy), status])
                if status == "ТРЕВОГА":
                    intrusion_detected = True
            else:
                table_data.append(["", "", "", "", ""])

        if not self.tables_are_equal(self.last_table_data, table_data):
            self.table.update_values(table_data)
            self.table.update()
            self.last_table_data = [row[:] for row in table_data]

        if intrusion_detected:
            self.status_label.configure(
                text="СТАТУС: ТРЕВОГА - НАРУШЕНИЕ ЗОНЫ!", text_color="#ff4444"
            )
            self.roi_alert_active = True
        elif self.roi_alert_active and not intrusion_detected:
            self.status_label.configure(text="СТАТУС: СЛЕЖЕНИЕ", text_color="#00ff00")
            self.roi_alert_active = False
        elif self.current_detections and not intrusion_detected:
            self.status_label.configure(
                text=f"СТАТУС: СЛЕЖЕНИЕ ({len(self.current_detections)} об.)",
                text_color="#00ff00",
            )
        elif not self.current_detections:
            self.status_label.configure(text="СТАТУС: ГОТОВ", text_color="#aaaaaa")

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                if self.video_source_type == "file":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_idx = 0
                    self.track_history.clear()
                    self.track_last_seen.clear()
                    self.track_class.clear()
                else:
                    self.status_label.configure(
                        text="СТАТУС: ОШИБКА ЗАХВАТА", text_color="#ff4444"
                    )
            else:
                self.current_frame_shape = frame.shape
                cam_h, cam_w = frame.shape[:2]

                frame = self.process_frame(frame)

                self.update_table()

                if self.drawing_roi:
                    x, y, w, h = map(int, self.roi_coords)
                    roi_color = (0, 0, 255) if self.roi_alert_active else (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), roi_color, 3)

                f_w = 600
                f_h = 700

                if cam_w > cam_h:
                    aspect = f_w / cam_w
                    tw, th = f_w, int(cam_h * aspect)
                else:
                    aspect = f_h / cam_h
                    tw, th = int(cam_w * aspect), f_h

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                ctk_img = ctk.CTkImage(
                    light_image=pil_img, dark_image=pil_img, size=(tw, th)
                )

                self.video_label.configure(image=ctk_img)
                self.video_label.image = ctk_img

        self.after(5, self.update_frame)

    def box_intersects_roi(self, x1, y1, x2, y2):
        rx, ry, rw, rh = self.roi_coords
        rx2, ry2 = rx + rw, ry + rh

        return not (x2 < rx or x1 > rx2 or y2 < ry or y1 > ry2)


if __name__ == "__main__":
    app = BurevestnikApp()
    app.mainloop()
