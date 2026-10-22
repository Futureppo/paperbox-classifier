"""Manual single-carton crops. This tool does not infer labels or import torch."""
import argparse
import csv
import hashlib
import json
import tkinter as tk
from collections import Counter
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageOps, ImageTk

FIELDS = ['path', 'label', 'group', 'source', 'box']


class Annotator:
    def __init__(self, root, sources, out, references):
        self.root, self.sources, self.out = root, sources.resolve(), out.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.csv_path, self.progress = self.out / 'annotations.csv', self.out / 'annotation_progress.json'
        self.frames = sorted(p for p in self.sources.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        if not self.frames:
            raise ValueError(f'No source images in {sources}')
        self.group_map = {}
        index = self.sources / 'index.csv'
        if index.exists():
            with index.open(encoding='utf-8-sig', newline='') as stream:
                self.group_map = {r['path']: r['group'] for r in csv.DictReader(stream)}
        self.rows = []
        if self.csv_path.exists():
            with self.csv_path.open(encoding='utf-8-sig', newline='') as stream:
                self.rows = list(csv.DictReader(stream))
        self.index, self.undo_rows, self.photo_refs, self.box = 0, [], [], None
        if self.progress.exists():
            previous = json.loads(self.progress.read_text(encoding='utf-8')).get('source', '')
            self.index = next((i for i, p in enumerate(self.frames)
                               if p.relative_to(self.sources).as_posix() == previous), 0)
        root.title('纸盒裁剪标注 · 拖框后按 0–5 保存 · A/D 上一张/下一张')
        top = tk.Frame(root)
        top.pack(fill='x')
        for c in range(6):
            frame = tk.Frame(top)
            frame.pack(side='left', padx=6)
            path = references / f'{c}.jpg'
            if path.exists():
                with Image.open(path) as image:
                    thumb = ImageOps.contain(image.convert('RGB'), (145, 90))
                photo = ImageTk.PhotoImage(thumb)
                self.photo_refs.append(photo)
                tk.Label(frame, image=photo).pack()
            tk.Button(frame, text=f'类别 {c} · 保存裁剪', command=lambda label=str(c): self.save(label)).pack()
        self.status = tk.StringVar()
        tk.Label(root, textvariable=self.status, justify='left').pack(fill='x')
        self.width = max(640, min(1160, root.winfo_screenwidth() - 80))
        self.height = max(350, min(650, root.winfo_screenheight() - 300))
        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg='#20252b', cursor='crosshair')
        self.canvas.pack()
        self.canvas.bind('<ButtonPress-1>', self.start)
        self.canvas.bind('<B1-Motion>', self.drag)
        self.canvas.bind('<ButtonRelease-1>', self.finish)
        buttons = tk.Frame(root)
        buttons.pack(fill='x', pady=8)
        tk.Button(buttons, text='上一张 [A]', command=lambda: self.move(-1)).pack(side='left')
        tk.Button(buttons, text='下一张 [D] / 跳过不确定图片', command=lambda: self.move(1)).pack(side='left')
        tk.Button(buttons, text='撤销本次会话最后一个裁剪 [U]', command=self.undo).pack(side='left')
        tk.Label(buttons, text='一个框只含一个完整盒子；模糊、遮挡严重或类别不确定时跳过。').pack(side='left', padx=12)
        for c in range(6):
            root.bind(str(c), lambda event, label=str(c): self.save(label))
        root.bind('a', lambda event: self.move(-1))
        root.bind('d', lambda event: self.move(1))
        root.bind('u', lambda event: self.undo())
        self.load()

    def load(self):
        self.path = self.frames[self.index]
        self.source = self.path.relative_to(self.sources).as_posix()
        self.group = self.group_map.get(self.source, self.path.parent.name)
        with Image.open(self.path) as image:
            self.original = ImageOps.exif_transpose(image).convert('RGB')
        displayed = ImageOps.contain(self.original, (self.width, self.height))
        self.sx = self.original.width / displayed.width
        self.sy = self.original.height / displayed.height
        self.display_width, self.display_height = displayed.size
        self.photo = ImageTk.PhotoImage(displayed)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.box = None
        self.progress.write_text(json.dumps({'source': self.source}, ensure_ascii=False), encoding='utf-8')
        self.refresh_status()

    def refresh_status(self):
        counts = Counter(r['label'] for r in self.rows)
        current = sum(r['source'] == self.source for r in self.rows)
        self.status.set(f'{self.index + 1}/{len(self.frames)} | {self.source} | 分组: {self.group} | 本图已保存 {current} 个\n'
                        + '各类累计: ' + ' / '.join(f'{i}: {counts[str(i)]}' for i in range(6)))

    def position(self, event):
        return max(0, min(event.x, self.display_width)), max(0, min(event.y, self.display_height))

    def start(self, event):
        self.start_point = self.position(event)
        self.box = None
        self.canvas.delete('selection')

    def drag(self, event):
        if not hasattr(self, 'start_point'):
            return
        x, y = self.position(event)
        self.canvas.delete('selection')
        self.canvas.create_rectangle(*self.start_point, x, y, outline='#00e5ff', width=3, tags='selection')

    def finish(self, event):
        if not hasattr(self, 'start_point'):
            return
        x, y = self.position(event)
        x0, y0 = self.start_point
        self.box = (round(min(x0, x) * self.sx), round(min(y0, y) * self.sy),
                    round(max(x0, x) * self.sx), round(max(y0, y) * self.sy))

    def write_csv(self):
        temporary = self.csv_path.with_suffix('.csv.tmp')
        with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        temporary.replace(self.csv_path)

    def save(self, label):
        if not self.box or self.box[2] - self.box[0] < 32 or self.box[3] - self.box[1] < 32:
            messagebox.showinfo('先画框', '请拖框选中一个完整盒子，再按类别 0–5。')
            return
        box_text = ','.join(map(str, self.box))
        if any(r['source'] == self.source and r['box'] == box_text for r in self.rows):
            messagebox.showinfo('已保存', '这个框已经保存。需要改类别时先撤销。')
            return
        identity = hashlib.sha256((self.source + ':' + box_text).encode()).hexdigest()[:20]
        relative = Path('crops') / label / f'{identity}.jpg'
        target = self.out / relative
        if target.exists():
            messagebox.showerror('文件冲突', f'已有裁剪文件，未覆盖：{target}')
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        self.original.crop(self.box).save(target, quality=95)
        row = dict(path=relative.as_posix(), label=label, group=self.group, source=self.source, box=box_text)
        self.rows.append(row)
        self.undo_rows.append(row)
        self.write_csv()
        self.box = None
        self.canvas.delete('selection')
        self.refresh_status()

    def undo(self):
        if not self.undo_rows:
            messagebox.showinfo('无法撤销', '这里只撤销本次打开工具后保存的裁剪。')
            return
        row = self.undo_rows.pop()
        self.rows.remove(row)
        target = (self.out / row['path']).resolve()
        if target.is_relative_to(self.out / 'crops'):
            target.unlink(missing_ok=True)
        self.write_csv()
        self.refresh_status()

    def move(self, offset):
        self.index = max(0, min(len(self.frames) - 1, self.index + offset))
        self.load()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, default=Path('source_frames'))
    parser.add_argument('--out', type=Path, default=Path('data'))
    parser.add_argument('--references', type=Path, default=Path('reference_labels'))
    args = parser.parse_args()
    root = tk.Tk()
    Annotator(root, args.sources, args.out, args.references)
    root.mainloop()


if __name__ == '__main__':
    main()
