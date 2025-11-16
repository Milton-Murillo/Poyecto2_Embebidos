#!/usr/bin/env python3
# car_video.py - Procesa un archivo de video.mp4 con un modelo TFLite tipo SSD
# y dibuja detecciones de vehículos. Termina cuando el video llega al final.

import argparse, sys, time, os
import numpy as np
import cv2

# Intérprete TFLite
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    from tensorflow.lite import Interpreter  # type: ignore

# COCO nombres por defecto 0-based
COCO80 = [
 "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
 "fire hydrant","stop sign","parking meter","bench","cow","bird","cat","dog","horse","sheep",
 "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
 "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
 "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
 "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
 "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
 "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]
VEHICLES = {"bicycle","car","motorcycle","bus","train","truck"}
ANIMALS  = {"cat","dog","horse","sheep","cow","bear"}
PERSON   = {"person"}

def load_labels(path: str|None):
    if not path or not os.path.exists(path): return None
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    m = {}
    for i, line in enumerate(lines):
        parts = line.split(maxsplit=1)
        if parts and parts[0].isdigit() and len(parts)==2:
            m[int(parts[0])] = parts[1]
        else:
            m[i] = line
    return m

def labels_to_list(lbl_dict):
    if not lbl_dict: return []
    n = max(lbl_dict.keys()) + 1
    return [lbl_dict.get(i, "???") for i in range(n)]

def resolve_name(cls_id: int, labels_list):
    # intenta labels externos 0 y 1 based; luego COCO80
    for off in (0, -1):
        idx = cls_id + off
        if 0 <= idx < len(labels_list):
            name = labels_list[idx]
            if name != "???":
                return name
    for off in (0, -1):
        idx = cls_id + off
        if 0 <= idx < len(COCO80):
            return COCO80[idx]
    return None

def quantize_if_needed(img_rgb, dtype, quant):
    scale, zero = quant if quant else (0.0, 0)
    if dtype == np.uint8:
        return img_rgb.astype(np.uint8)
    if dtype == np.int8:
        if scale == 0: scale = 1.0/255.0
        x = img_rgb.astype(np.float32)/255.0
        x = x/scale + zero
        return np.clip(np.rint(x), -128, 127).astype(np.int8)
    return img_rgb.astype(np.float32)/255.0

def squeeze01(x):
    x = np.array(x)
    if x.ndim>=3 and x.shape[0]==1: return np.squeeze(x, axis=0)
    if x.ndim>=2 and x.shape[0]==1: return np.squeeze(x, axis=0)
    return x

def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = area_a + area_b - inter + 1e-9
    return inter / union

def nms_xyxy(boxes, scores, iou_thr):
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while idxs:
        i = idxs.pop(0)
        keep.append(i)
        idxs = [j for j in idxs if iou_xyxy(boxes[i], boxes[j]) < iou_thr]
    return keep

# ================== util de UI para 2 fuentes ==================
def stack_frames_h(f_left, f_right):
    if f_left is None and f_right is None: return None
    if f_left is None: return f_right.copy()
    if f_right is None: return f_left.copy()
    h = min(f_left.shape[0], f_right.shape[0])
    if f_left.shape[0] != h:
        f_left  = cv2.resize(f_left,  (int(f_left.shape[1]*h/f_left.shape[0]), h))
    if f_right.shape[0] != h:
        f_right = cv2.resize(f_right, (int(f_right.shape[1]*h/f_right.shape[0]), h))
    return np.hstack([f_left, f_right])

def put_banner(img, persons, animals, vehicles, color_txt, tleft_sec):
    font=cv2.FONT_HERSHEY_SIMPLEX
    s=0.8; th=2
    if tleft_sec is None:
        tstr="--:--"
    else:
        tleft_sec=max(0,int(tleft_sec))
        m=int(tleft_sec//60); s_left=int(tleft_sec%60); tstr=f"{m:02d}:{s_left:02d}"
    text=f"Personas: {persons} | Animales: {animals} | Vehiculos: {vehicles}   |   Semaforo: {color_txt}   Tiempo: {tstr}"
    (tw,th_text), _ = cv2.getTextSize(text, font, s, th)
    pad_x, pad_y = 10, 10
    w = min(tw+pad_x*2, img.shape[1]-1)
    cv2.rectangle(img,(0,0),(w, th_text+pad_y*2),(0,0,0),-1)
    cv2.putText(img, text, (pad_x, th_text+pad_y-2), font, s, (255,255,255), th, cv2.LINE_AA)

# ================== semáforo con histeresis ==================
class TL:
    GREEN="GREEN"
    YELLOW="YELLOW"
    RED_TIMED="RED_TIMED"
    RED_HOLD="RED_HOLD"

class TrafficLightFSM:
    def __init__(self, hold_sec=5.0):
        self.H = float(hold_sec)
        self.state=TL.RED_HOLD
        self.until=0.0
        self.last_ts=time.time()
        self.h_vehicle=0.0
        self.h_person =0.0
        self.h_animal =0.0
        # escalones estables de personas
        self.green_applied_persons=0
        self.green_stable_persons=0
        self.green_cand_persons=0
        self.green_cand_time=0.0
        # aviso visual
        self._notice_txt=""
        self._notice_t0=0.0
        self._notice_dur=2.0  # segundos de aviso

    def _dt(self):
        now=time.time()
        dt=max(0.0, now-self.last_ts)
        self.last_ts=now
        return now, dt

    def _apply_hold(self, present, hold, dt):
        if present: hold=min(self.H, hold+dt)
        else:       hold=max(0.0, hold-dt)
        return hold

    def _enter_green(self, now):
        self.state=TL.GREEN
        self.until=now + 180.0
        self.green_applied_persons=0
        self.green_stable_persons=0
        self.green_cand_persons=0
        self.green_cand_time=0.0

    def _enter_yellow(self, now):
        self.state=TL.YELLOW
        self.until=now + 5.0

    def _enter_red_timed(self, now, sec=60.0):
        self.state=TL.RED_TIMED
        self.until=now + sec

    def _enter_red_hold(self):
        self.state=TL.RED_HOLD
        self.until=0.0

    def _notice(self, txt):
        self._notice_txt = txt
        self._notice_t0  = time.time()

    def get_notice(self):
        if self._notice_txt and (time.time() - self._notice_t0) <= self._notice_dur:
            return self._notice_txt
        return ""

    def _update_stable_persons(self, persons_total, dt):
        # Reglas: solo aplica en VERDE, con presencia sostenida >= H
        if persons_total <= self.green_stable_persons:
            self.green_cand_persons = 0
            self.green_cand_time = 0.0
            return 0
        if persons_total != self.green_cand_persons:
            self.green_cand_persons = persons_total
            self.green_cand_time = 0.0
            return 0
        self.green_cand_time += dt
        if self.green_cand_time >= self.H:
            new_people = self.green_cand_persons - self.green_stable_persons
            self.green_stable_persons = self.green_cand_persons
            self.green_cand_persons = 0
            self.green_cand_time = 0.0
            return max(0, new_people)
        return 0

    def update(self, persons_total, animals_total, vehicles_total):
        now, dt = self._dt()
        # histeresis
        self.h_vehicle = self._apply_hold(vehicles_total>0, self.h_vehicle, dt)
        self.h_person  = self._apply_hold(persons_total>0,  self.h_person,  dt)
        self.h_animal  = self._apply_hold(animals_total>0,  self.h_animal,  dt)

        veh_ok = (self.h_vehicle >= self.H)
        veh_abs = (self.h_vehicle <= 0.0)
        per_ok = (self.h_person  >= self.H)
        ani_ok = (self.h_animal  >= self.H)

        if veh_abs:
            self._enter_red_hold()
        else:
            if self.state==TL.RED_HOLD and veh_ok:
                self._enter_red_timed(now, 60.0)

            if self.state==TL.GREEN:
                # Penalización solo por personas nuevas sostenidas >= H
                if per_ok:
                    new_people = self._update_stable_persons(persons_total, dt)
                    if new_people > 0:
                        self.until -= 30.0 * new_people
                        self.green_applied_persons += new_people
                        self._notice(f"-30 s x {new_people} persona(s)")
                # Animal estable => Amarillo inmediato
                if ani_ok:
                    self._enter_yellow(now)
                elif now >= self.until:
                    self._enter_yellow(now)

            elif self.state==TL.YELLOW:
                if now >= self.until:
                    self._enter_red_timed(now, 60.0)

            elif self.state==TL.RED_TIMED:
                if now >= self.until:
                    self._enter_green(now)

            else:
                if veh_ok:
                    self._enter_red_timed(now, 60.0)

        if self.state==TL.GREEN:
            color_txt="Verde"; tleft=self.until - now
        elif self.state==TL.YELLOW:
            color_txt="Amarillo"; tleft=self.until - now
        elif self.state==TL.RED_TIMED:
            color_txt="Rojo"; tleft=self.until - now
        else:
            color_txt="Rojo"; tleft=None
        return color_txt, tleft

# ================== helpers de UI ==================
def stack_frames_h(f_left, f_right):
    if f_left is None and f_right is None: return None
    if f_left is None: return f_right.copy()
    if f_right is None: return f_left.copy()
    h = min(f_left.shape[0], f_right.shape[0])
    if f_left.shape[0] != h:
        f_left  = cv2.resize(f_left,  (int(f_left.shape[1]*h/f_left.shape[0]), h))
    if f_right.shape[0] != h:
        f_right = cv2.resize(f_right, (int(f_right.shape[1]*h/f_right.shape[0]), h))
    return np.hstack([f_left, f_right])

def draw_notice(img, txt):
    if not txt: return
    y = 88
    cv2.putText(img, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser(description="Procesar video.mp4 con un modelo TFLite SSD y detectar vehículos.")
    ap.add_argument("--model", required=True, help="Ruta al .tflite")
    ap.add_argument("--labels", default=None, help="Opcional: labels.txt")
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--nms_iou", type=float, default=0.50)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--min_area_ratio", type=float, default=0.003, help="área mínima relativa para dibujar cajas")
    ap.add_argument("--show_fps", action="store_true", default=True)
    args = ap.parse_args()

    # Intérprete
    interpreter = Interpreter(model_path=args.model, num_threads=max(1, args.threads))
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()
    ih, iw = int(in_det["shape"][1]), int(in_det["shape"][2])

    # Labels
    labels_map = load_labels(args.labels)
    labels_list = labels_to_list(labels_map) if labels_map else []

    # Fuente 1: video para VEHÍCULOS
    cap_vid = cv2.VideoCapture('video.mp4')
    if not cap_vid.isOpened():
        print("No pude abrir 'video.mp4'. Verifica la ruta o el códec.", file=sys.stderr)
        sys.exit(1)

    # Fuente 2: cámara para PERSONAS y ANIMALES
    cap_cam = cv2.VideoCapture(0)
    if not cap_cam.isOpened():
        print("ADVERTENCIA: no pude abrir la cámara 0. Seguiré solo con el video.", file=sys.stderr)
        cap_cam = None

    win = "Detección combinada"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    prev = time.time()
    fps = 0.0
    fsm = TrafficLightFSM(hold_sec=5.0)

    while True:
        # Leer video
        okv, frame_v = cap_vid.read()
        if not okv:
            break
        Hv, Wv = frame_v.shape[:2]
        area_min_v = args.min_area_ratio * Wv * Hv

        # Inferencia vehículos en frame_v
        frame_rgb = cv2.cvtColor(frame_v, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)
        inp = quantize_if_needed(resized, in_det["dtype"], in_det.get("quantization"))
        interpreter.set_tensor(in_det["index"], np.expand_dims(inp, 0))
        interpreter.invoke()

        outs = [interpreter.get_tensor(o['index']) for o in out_det]
        if len(outs) >= 4:
            boxes_n  = squeeze01(outs[0]); classes_v= squeeze01(outs[1]).astype(np.int32)
            scores_v = squeeze01(outs[2]).astype(np.float32); num = int(np.squeeze(outs[3]))
        elif len(outs) == 3:
            boxes_n  = squeeze01(outs[0]); classes_v= squeeze01(outs[1]).astype(np.int32)
            scores_v = squeeze01(outs[2]).astype(np.float32); num = min(len(scores_v), len(classes_v), len(boxes_n))
        else:
            print("El modelo no parece SSD estándar.", file=sys.stderr); break

        cand_boxes_v, cand_scores_v, cand_disp_v = [], [], []
        for i in range(num):
            s = float(scores_v[i])
            if s < args.threshold: continue
            cls_id = int(classes_v[i]); name = resolve_name(cls_id, labels_list)
            if not name or name.lower() not in VEHICLES: continue
            y1,x1,y2,x2 = boxes_n[i]
            x1p, y1p = int(x1*Wv), int(y1*Hv); x2p, y2p = int(x2*Wv), int(y2*Hv)
            if (x2p-x1p)*(y2p-y1p) < area_min_v: continue
            cand_boxes_v.append([x1p,y1p,x2p,y2p]); cand_scores_v.append(s); cand_disp_v.append(f"{name} {s:.2f}")

        keep_v = nms_xyxy(cand_boxes_v, cand_scores_v, args.nms_iou) if cand_boxes_v else []
        vehicles_total = 0
        for k in keep_v:
            x1,y1,x2,y2 = cand_boxes_v[k]
            cv2.rectangle(frame_v, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame_v, cand_disp_v[k], (x1, max(0,y1-7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,255,0), 2, cv2.LINE_AA)
            vehicles_total += 1

        # Leer cámara y detectar personas/animales
        persons_total = animals_total = 0
        frame_c = None
        if cap_cam is not None:
            okc, frame_c = cap_cam.read()
            if okc:
                Hc, Wc = frame_c.shape[:2]
                area_min_c = args.min_area_ratio * Wc * Hc
                rgb_c = cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB)
                resized_c = cv2.resize(rgb_c, (iw, ih), interpolation=cv2.INTER_LINEAR)
                inp_c = quantize_if_needed(resized_c, in_det["dtype"], in_det.get("quantization"))
                interpreter.set_tensor(in_det["index"], np.expand_dims(inp_c, 0))
                interpreter.invoke()
                outs_c = [interpreter.get_tensor(o['index']) for o in out_det]
                if len(outs_c) >= 4:
                    boxes_c  = squeeze01(outs_c[0]); classes_c= squeeze01(outs_c[1]).astype(np.int32)
                    scores_c = squeeze01(outs_c[2]).astype(np.float32); num_c = int(np.squeeze(outs_c[3]))
                elif len(outs_c) == 3:
                    boxes_c  = squeeze01(outs_c[0]); classes_c= squeeze01(outs_c[1]).astype(np.int32)
                    scores_c = squeeze01(outs_c[2]).astype(np.float32); num_c = min(len(scores_c), len(classes_c), len(boxes_c))
                else:
                    boxes_c=classes_c=scores_c=[]; num_c=0

                cand_boxes_c, cand_scores_c, cand_disp_c = [], [], []
                for i in range(num_c):
                    s = float(scores_c[i])
                    if s < args.threshold: continue
                    cls_id = int(classes_c[i]); name = resolve_name(cls_id, labels_list)
                    nl = (name or "").lower()
                    if nl not in PERSON and nl not in ANIMALS: continue
                    y1,x1,y2,x2 = boxes_c[i]
                    x1p, y1p = int(x1*Wc), int(y1*Hc); x2p, y2p = int(x2*Wc), int(y2*Hc)
                    if (x2p-x1p)*(y2p-y1p) < area_min_c: continue
                    cand_boxes_c.append([x1p,y1p,x2p,y2p]); cand_scores_c.append(s); cand_disp_c.append((nl, f"{name} {s:.2f}"))

                keep_c = nms_xyxy(cand_boxes_c, cand_scores_c, args.nms_iou) if cand_boxes_c else []
                for k in keep_c:
                    x1,y1,x2,y2 = cand_boxes_c[k]
                    nl, txt = cand_disp_c[k]
                    if nl in PERSON:
                        persons_total += 1; color=(255,255,0)
                    else:
                        animals_total += 1; color=(255,0,255)
                    cv2.rectangle(frame_c, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(frame_c, txt, (x1, max(0,y1-7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

        # Actualizar FSM y banner
        color_txt, tleft = fsm.update(persons_total, animals_total, vehicles_total)
        canvas = stack_frames_h(frame_v, frame_c if frame_c is not None else None)
        if canvas is None: canvas = frame_v
        put_banner(canvas, persons_total, animals_total, vehicles_total, color_txt, tleft)
        draw_notice(canvas, fsm.get_notice())  # muestra "-30 s x N persona(s)" por 2 s

        # FPS
        if args.show_fps:
            now = time.time()
            fps = 0.9*fps + 0.1*(1.0/max(1e-6, now - prev))
            prev = now
 
            h = canvas.shape[0]
            y = h - 10  # margen inferior
 
            cv2.putText(canvas, f"FPS {fps:.1f}", (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(canvas, f"FPS {fps:.1f}", (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow(win, canvas)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap_vid.release()
    if cap_cam is not None:
        cap_cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

