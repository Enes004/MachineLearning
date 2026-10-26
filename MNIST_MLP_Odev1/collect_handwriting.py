"""A small drawing app for collecting the student's own handwritten digits."""

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, simpledialog

from PIL import Image, ImageDraw

from handwriting_utils import normalize_handwriting_image


class DigitCollector:
    def __init__(self, student_number: str, output_dir: Path) -> None:
        self.student_number = student_number
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.digits = list(student_number)
        self.position = 0
        self.root = tk.Tk()
        self.root.title("Kendi El Yazinizla Rakam Toplama")
        self.root.resizable(False, False)
        self.canvas_size = 280
        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_size,
            height=self.canvas_size,
            background="white",
            cursor="cross",
        )
        self.canvas.pack(padx=12, pady=(12, 4))
        self.status = tk.Label(self.root, font=("Arial", 12))
        self.status.pack(pady=6)
        buttons = tk.Frame(self.root)
        buttons.pack(pady=(0, 12))
        tk.Button(buttons, text="Temizle", command=self.clear).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons, text="Kaydet ve devam et", command=self.save_digit).pack(
            side=tk.LEFT, padx=5
        )

        self.canvas.bind("<Button-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.draw_stroke)
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.drawer = ImageDraw.Draw(self.image)
        self.last_point: tuple[int, int] | None = None
        self.update_status()

    def update_status(self) -> None:
        digit = self.digits[self.position]
        self.status.config(
            text=f"Okul numaranizdaki {self.position + 1}/{len(self.digits)}. rakam: {digit}"
        )

    def start_stroke(self, event: tk.Event) -> None:
        self.last_point = (event.x, event.y)
        radius = 9
        self.drawer.ellipse(
            (event.x - radius, event.y - radius, event.x + radius, event.y + radius),
            fill=0,
        )
        self.canvas.create_oval(
            event.x - radius,
            event.y - radius,
            event.x + radius,
            event.y + radius,
            fill="black",
            outline="black",
        )

    def draw_stroke(self, event: tk.Event) -> None:
        if self.last_point is None:
            return
        point = (event.x, event.y)
        self.drawer.line((self.last_point, point), fill=0, width=18)
        self.drawer.ellipse((event.x - 9, event.y - 9, event.x + 9, event.y + 9), fill=0)
        self.canvas.create_line(
            *self.last_point, event.x, event.y, fill="black", width=18, capstyle=tk.ROUND
        )
        self.last_point = point

    def clear(self) -> None:
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.drawer = ImageDraw.Draw(self.image)
        self.last_point = None

    def save_digit(self) -> None:
        expected = self.digits[self.position]
        filename = f"digit_{self.position + 1:02d}_label_{expected}.png"
        try:
            normalized = normalize_handwriting_image(self.image)
        except ValueError as error:
            messagebox.showerror("Bos rakam", str(error), parent=self.root)
            return
        normalized.save(self.output_dir / filename)
        self.position += 1
        if self.position == len(self.digits):
            messagebox.showinfo(
                "Tamamlandi",
                f"{len(self.digits)} rakam kaydedildi:\n{self.output_dir.resolve()}",
                parent=self.root,
            )
            self.root.destroy()
            return
        self.clear()
        self.update_status()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/handwriting"),
        help="Directory for 28x28 digit images (default: data/handwriting).",
    )
    args = parser.parse_args()
    root = tk.Tk()
    root.withdraw()
    student_number = simpledialog.askstring(
        "Okul numarasi",
        "Okul numaranizi girin. Her rakami ayri ayri kendi el yazinizla cizin:",
        parent=root,
    )
    root.destroy()
    if student_number is None:
        return
    if not student_number or not student_number.isdigit():
        parser.error("School number must contain digits only.")
    DigitCollector(student_number, args.output_dir).run()


if __name__ == "__main__":
    main()
