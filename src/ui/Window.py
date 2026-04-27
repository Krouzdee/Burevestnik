import customtkinter as ctk
from CTkTable import CTkTable
import cv2
from PIL import Image

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


def center(work, x: int, y: int) -> str:
    POS_X = work.winfo_screenwidth() // 2 - x // 2
    POS_Y = work.winfo_screenheight() // 2 - y // 2
    return f"{x}x{y}+{POS_X}+{POS_Y}"


class BurevestnikApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("БУРЕВЕСТНИК - Система мониторинга БПЛА")
        self.geometry(center(self, 1280, 720))
        self.resizable(width=False, height=False)

        self.cap = None
        self.drawing_roi = False
        self.roi_drawing_active = False
        self.roi_start_x = 0
        self.roi_start_y = 0
        self.roi_coords = [100, 100, 300, 300]

        self.current_frame_shape = (0, 0)  # h, w

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

        self.cam_select = ctk.CTkOptionMenu(
            self.control_panel,
            values=self.get_available_cameras(),
            command=self.change_camera,
        )
        self.cam_select.place(x=30, y=80, relwidth=0.9)

        self.btn_roi = ctk.CTkButton(
            self.control_panel,
            text="Включить ROI",
            height=40,
            fg_color="#2b7b50",
            command=self.toggle_roi_mode,
        )
        self.btn_roi.place(x=30, y=130, relwidth=0.9)

        self.table = CTkTable(
            master=self.control_panel,
            row=6,
            column=4,
            values=[["ID", "Тип", "X", "Y"]] + [[""] * 4] * 5,
        )
        self.table.place(x=30, y=200, relwidth=0.9)

        self.status_label = ctk.CTkLabel(
            self.control_panel,
            text="СТАТУС: ГОТОВ",
            font=("Arial", 13, "bold"),
            text_color="#aaaaaa",
        )
        self.status_label.place(relx=0.5, rely=0.95, anchor="center")

        if self.cam_select.get() != "Камеры не найдены":
            self.change_camera(self.cam_select.get())

        self.update_frame()

    def _event_to_frame_coords(self, event):
        cam_h, cam_w = self.current_frame_shape[:2]
        if cam_w == 0:
            return None, None

        display_w = self.video_label.winfo_width()
        display_h = self.video_label.winfo_height()

        scale_x = cam_w / display_w
        scale_y = cam_h / display_h

        return event.x * scale_x, event.y * scale_y

    def on_mouse_down(self, event):
        if not self.drawing_roi or self.cap is None:
            return
        fx, fy = self._event_to_frame_coords(event)
        self.roi_drawing_active = True
        self.roi_start_x, self.roi_start_y = fx, fy

    def on_mouse_move(self, event):
        if not self.drawing_roi or not self.roi_drawing_active:
            return
        fx, fy = self._event_to_frame_coords(event)

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
        txt = " Выключить ROI" if self.drawing_roi else "Включить ROI"
        self.btn_roi.configure(text=txt, fg_color=color)

    def get_available_cameras(self):
        cams = []
        for i in range(3):
            temp = cv2.VideoCapture(i)
            if temp.isOpened():
                cams.append(f"Камера {i}")
                temp.release()
        return cams if cams else ["Камеры не найдены"]

    def change_camera(self, choice):
        if self.cap:
            self.cap.release()
        try:
            idx = int(choice.split()[-1])
            self.cap = cv2.VideoCapture(idx)
        except:
            pass

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame_shape = frame.shape
                cam_h, cam_w = frame.shape[:2]

                if self.drawing_roi:
                    x, y, w, h = map(int, self.roi_coords)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"ROI: {w}x{h}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                f_w = self.video_frame.winfo_width()
                f_h = self.video_frame.winfo_height()

                if f_w > 10 and f_h > 10:
                    aspect = cam_w / cam_h
                    tw, th = f_w, int(f_w / aspect)
                    if th > f_h:
                        th, tw = f_h, int(f_h * aspect)

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)

                    ctk_img = ctk.CTkImage(
                        light_image=pil_img, dark_image=pil_img, size=(tw, th)
                    )

                    self.video_label.configure(image=ctk_img)
                    self.video_label.image = ctk_img

        self.after(15, self.update_frame)


if __name__ == "__main__":
    app = BurevestnikApp()
    app.mainloop()
