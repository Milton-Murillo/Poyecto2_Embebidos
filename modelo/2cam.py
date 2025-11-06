#!/usr/bin/env python3
# 2cam_hold_stable.py — Dos cámaras en una ventana; TFLite; conteos y semáforo SOLO texto.
# Histeresis configurable (--hold_sec, por defecto 5 s) para considerar personas/animales/vehículos.
# Flujo normal: Verde 3min -> Amarillo 5s -> Rojo 1min -> Verde...
# Personas (en Verde): -30s por persona NUEVA solo si el nuevo conteo se sostiene >= hold_sec.
# Animales (en Verde): cambio a Amarillo solo si animal sostenido >= hold_sec; luego Rojo 1min.
# Vehículos: ausencia sostenida => Rojo HOLD (sin tiempo). Presencia sostenida saliendo de HOLD => Rojo 1min.

import argparse, time, os, sys, threading
import numpy as np, cv2

# ---------- Intérprete TFLite ----------
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    from tensorflow.lite import Interpreter  # type: ignore

# ---------- COCO fallback (0-based) ----------
COCO80 = [
 "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
 "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
 "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
 "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
 "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
 "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
 "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
 "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]
VEHICLES = {"bicycle","car","motorcycle","bus","train","truck"}
ANIMALS  = {"cat","dog","horse","sheep","cow","bear"}
MACROS   = {"vehicles":VEHICLES, "animals":ANIMALS, "person":{"person"}, "person_animals":{"person"}|ANIMALS}

# ---------- Labels opcionales ----------
def load_labels(path):
    if not path or not os.path.exists(path): return None
    with open(path,"r",encoding="utf-8") as f:
        lines=[l.strip() for l in f if l.strip()]
    m={}
    for i,line in enumerate(lines):
        parts=line.split(maxsplit=1)
        if parts and parts[0].isdigit() and len(parts)==2: m[int(parts[0])]=parts[1]
        else: m[i]=line
    return m

def labels_to_list(lbl):
    if not lbl: return []
    n=max(lbl.keys())+1
    return [lbl.get(i,"???") for i in range(n)]

def resolve_name(cls_id, labels_list):
    for off in (0,-1):
        idx=cls_id+off
        if 0<=idx<len(labels_list):
            name=labels_list[idx]
            if name!="???": return name, off
    for off in (0,-1):
        idx=cls_id+off
        if 0<=idx<len(COCO80): return COCO80[idx], off
    return None, None

# ---------- Pre/Post ----------
def quantize_if_needed(img_rgb, dtype, quant):
    scale, zero = quant if quant else (0.0, 0)
    if dtype==np.uint8: return img_rgb.astype(np.uint8)
    if dtype==np.int8:
        if scale==0: scale=1.0/255.0
        x=img_rgb.astype(np.float32)/255.0
        x=x/scale + zero
        return np.clip(np.rint(x), -128, 127).astype(np.int8)
    return img_rgb.astype(np.float32)/255.0

def squeeze01(x):
    x=np.array(x)
    if x.ndim>=3 and x.shape[0]==1: return np.squeeze(x,0)
    if x.ndim>=2 and x.shape[0]==1: return np.squeeze(x,0)
    return x

def iou_xyxy(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1])
    x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    area_a=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    area_b=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    union=area_a+area_b-inter+1e-9
    return inter/union

def nms_xyxy(boxes,scores,iou_thr):
    idxs=sorted(range(len(scores)), key=lambda i:scores[i], reverse=True)
    keep=[]
    while idxs:
        i=idxs.pop(0); keep.append(i)
        idxs=[j for j in idxs if iou_xyxy(boxes[i],boxes[j])<iou_thr]
    return keep

# ---------- Filtros ----------
def parse_keep(expr, default_names):
    names=set(default_names) if not expr else set()
    ids0, ids1=set(), set()
    if not expr: return names, ids0, ids1
    for t in [t.strip() for t in expr.split(",") if t.strip()]:
        tl=t.lower()
        if tl.startswith("@"):
            names |= {x.lower() for x in MACROS.get(tl[1:], set())}
        elif tl.startswith("ids1:"):
            ids1 |= set(int(x) for x in tl.split(":",1)[1].split(",") if x)
        elif tl.startswith("ids:"):
            ids0 |= set(int(x) for x in tl.split(":",1)[1].split(",") if x)
        else:
            names.add(tl)
    return names, ids0, ids1

def allowed(name, cls_id, names_set, ids0_set, ids1_set):
    by_name=(name is not None) and (name.lower() in names_set)
    by_id=(cls_id in ids0_set) or (cls_id in ids1_set)
    return by_name or by_id

def categorize(name):
    nl=(name or "").lower()
    if nl=="person": return "person"
    if nl in ANIMALS: return "animal"
    if nl in VEHICLES: return "vehicle"
    return "other"

# ---------- Captura baja latencia ----------
class LatestFrame:
    def __init__(self, cap):
        self.cap=cap; self.lock=threading.Lock()
        self.frame=None; self.ok=False; self.stop=False
        self.t=threading.Thread(target=self._loop, daemon=True); self.t.start()
    def _loop(self):
        while not self.stop:
            ok,f=self.cap.read()
            if not ok: time.sleep(0.005); continue
            with self.lock:
                self.ok=True; self.frame=f
    def read(self):
        with self.lock:
            return self.ok, None if self.frame is None else self.frame
    def close(self):
        self.stop=True; self.t.join(timeout=0.2)

def open_capture(src_str,w,h,fourcc):
    def try_open(arg,api=0):
        cap=cv2.VideoCapture(arg,api)
        if cap.isOpened():
            is_v4l=isinstance(arg,int) or (isinstance(arg,str) and str(arg).startswith("/dev/video"))
            if is_v4l:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT,h)
                cap.set(cv2.CAP_PROP_FPS,30)
                try: cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
                except: pass
                try:
                    if fourcc and fourcc.upper()!="ANY":
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc.upper()))
                    else:
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                except: pass
            return cap
        return None
    cap=None
    if src_str.isdigit():
        idx=int(src_str)
        cap = try_open(idx, cv2.CAP_V4L2) or try_open(idx, cv2.CAP_ANY) \
            or try_open(f"/dev/video{idx}", cv2.CAP_V4L2) or try_open(f"/dev/video{idx}", cv2.CAP_ANY)
    else:
        cap = try_open(src_str, cv2.CAP_V4L2) or try_open(src_str, cv2.CAP_ANY)
    if not cap: raise RuntimeError(f"No pude abrir {src_str}")
    return cap

# ---------- UI ----------
def stack_frames(f0,f1,layout='h'):
    if f0 is None and f1 is None: return None
    if f0 is None: return f1.copy()
    if f1 is None: return f0.copy()
    if layout=='h':
        h=min(f0.shape[0], f1.shape[0])
        if f0.shape[0]!=h: f0=cv2.resize(f0,(int(f0.shape[1]*h/f0.shape[0]),h))
        if f1.shape[0]!=h: f1=cv2.resize(f1,(int(f1.shape[1]*h/f1.shape[0]),h))
        return np.hstack([f0,f1])
    w=min(f0.shape[1], f1.shape[1])
    if f0.shape[1]!=w: f0=cv2.resize(f0,(w,int(f0.shape[0]*w/f0.shape[1])))
    if f1.shape[1]!=w: f1=cv2.resize(f1,(w,int(f1.shape[0]*w/f1.shape[1])))
    return np.vstack([f0,f1])

def put_banner(img, persons, animals, vehicles, color_txt, tleft_sec):
    font=cv2.FONT_HERSHEY_SIMPLEX
    s=0.8; th=2
    if tleft_sec is None:
        tstr="--:--"
    else:
        tleft_sec=max(0,int(tleft_sec))
        m=int(tleft_sec//60); s_left=int(tleft_sec%60); tstr=f"{m:02d}:{s_left:02d}"
    text=f"Personas: {persons} | Animales: {animals} | Vehículos: {vehicles}   |   Semáforo: {color_txt}   Tiempo: {tstr}"
    (tw,th_text), _ = cv2.getTextSize(text, font, s, th)
    pad_x, pad_y = 10, 10
    w = min(tw+pad_x*2, img.shape[1]-1)
    cv2.rectangle(img,(0,0),(w, th_text+pad_y*2),(0,0,0),-1)
    cv2.putText(img, text, (pad_x, th_text+pad_y-2), font, s, (255,255,255), th, cv2.LINE_AA)

# ---------- Semáforo con histeresis y penalización estable ----------
class TL:
    GREEN="GREEN"           # Verde con countdown
    YELLOW="YELLOW"         # Amarillo 5s
    RED_TIMED="RED_TIMED"   # Rojo 1min con countdown
    RED_HOLD="RED_HOLD"     # Rojo sin tiempo (ausencia sostenida de vehículos)

class TrafficLightFSM:
    def __init__(self, hold_sec=5.0):
        self.H = float(hold_sec)
        self.state=TL.RED_HOLD
        self.until=0.0
        self.last_ts=time.time()
        # holds (suben hasta H si hay presencia; bajan a 0 si no)
        self.h_vehicle=0.0
        self.h_person =0.0
        self.h_animal =0.0
        # VERDE: control de “escalones” estables de personas
        self.green_applied_persons=0      # penalizaciones ya aplicadas
        self.green_stable_persons=0       # escalón estable consolidado
        self.green_cand_persons=0         # candidato a nuevo escalón
        self.green_cand_time=0.0          # tiempo que se sostiene el candidato

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
        self.until=now + 180.0  # 3 min
        # reset de escalones/personas para este nuevo VERDE
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

    def _update_stable_persons(self, persons_total, dt):
        """
        Gestiona “escalones” de personas: solo cuando un NUEVO conteo
        >= (green_stable_persons + 1) se sostiene >= H, se consolida
        y se aplican -30s por cada nueva persona del escalón.
        """
        # Si baja por debajo del estable, ignoramos (no devolvemos tiempo).
        if persons_total <= self.green_stable_persons:
            self.green_cand_persons = 0
            self.green_cand_time = 0.0
            return 0  # delta de personas nuevas consolidadas

        # Hay un posible nuevo escalón
        if persons_total != self.green_cand_persons:
            self.green_cand_persons = persons_total
            self.green_cand_time = 0.0
            return 0

        # Mismo candidato, acumular tiempo
        self.green_cand_time += dt
        if self.green_cand_time >= self.H:
            # Consolidar escalón y devolver cuántas personas nuevas son
            new_people = self.green_cand_persons - self.green_stable_persons
            self.green_stable_persons = self.green_cand_persons
            # reset candidato para observar un posible siguiente escalón
            self.green_cand_persons = 0
            self.green_cand_time = 0.0
            return max(0, new_people)
        return 0

    def update(self, persons_total, animals_total, vehicles_total):
        now, dt = self._dt()

        # Histeresis de presencia/ausencia
        self.h_vehicle = self._apply_hold(vehicles_total>0, self.h_vehicle, dt)
        self.h_person  = self._apply_hold(persons_total>0,  self.h_person,  dt)
        self.h_animal  = self._apply_hold(animals_total>0,  self.h_animal,  dt)

        veh_ok = (self.h_vehicle >= self.H)   # presencia sostenida
        veh_abs = (self.h_vehicle <= 0.0)     # ausencia sostenida (~>=H sin vehículos)
        per_ok = (self.h_person  >= self.H)
        ani_ok = (self.h_animal  >= self.H)

        # Ausencia sostenida de vehículos => rojo HOLD
        if veh_abs:
            self._enter_red_hold()
        else:
            # Si veníamos de HOLD y ya hay vehículos sostenidos => Rojo 1 min
            if self.state==TL.RED_HOLD and veh_ok:
                self._enter_red_timed(now, 60.0)

            if self.state==TL.GREEN:
                # Penalización solo por PERSONAS ESTABLES (>=H) y por escalones consolidados
                if per_ok:
                    new_people = self._update_stable_persons(persons_total, dt)
                    if new_people > 0:
                        self.until -= 30.0 * new_people
                        self.green_applied_persons += new_people

                # Animal estable => inmediato a Amarillo
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
                # Seguridad
                if veh_ok:
                    self._enter_red_timed(now, 60.0)

        # Salida textual
        if self.state==TL.GREEN:
            color_txt="Verde"; tleft=self.until - now
        elif self.state==TL.YELLOW:
            color_txt="Amarillo"; tleft=self.until - now
        elif self.state==TL.RED_TIMED:
            color_txt="Rojo"; tleft=self.until - now
        else:  # RED_HOLD
            color_txt="Rojo"; tleft=None
        return color_txt, tleft

# ---------- Main ----------
def main():
    ap=argparse.ArgumentParser(description="Dos cámaras, una ventana; conteos y semáforo SOLO texto (histeresis estable).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--cam0", required=True)
    ap.add_argument("--cam1", required=True)
    ap.add_argument("--cap_w", type=int, default=640)
    ap.add_argument("--cap_h", type=int, default=480)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--nms_iou", type=float, default=0.50)
    ap.add_argument("--min_area_ratio", type=float, default=0.01)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--threads0", type=int)
    ap.add_argument("--threads1", type=int)
    ap.add_argument("--cam0_keep", default="@vehicles")
    ap.add_argument("--cam1_keep", default="@person_animals")
    ap.add_argument("--infer_every", type=int, default=1)
    ap.add_argument("--fourcc", default="MJPG", help="MJPG, YUYV, H264 o ANY")
    ap.add_argument("--layout", choices=["h","v"], default="h")
    ap.add_argument("--hold_sec", type=float, default=5.0, help="Margen (s) para confirmar presencia/ausencia")
    # ROI y overrides
    ap.add_argument("--crop0_top", type=float, default=0.0)
    ap.add_argument("--crop0_bottom", type=float, default=1.0)
    ap.add_argument("--crop1_top", type=float, default=0.0)
    ap.add_argument("--crop1_bottom", type=float, default=1.0)
    ap.add_argument("--thr0", type=float)
    ap.add_argument("--thr1", type=float)
    ap.add_argument("--mar0", type=float)
    ap.add_argument("--mar1", type=float)
    ap.add_argument("--topk0", type=int, default=50)
    ap.add_argument("--topk1", type=int, default=50)
    ap.add_argument("--hide_fps", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args=ap.parse_args()

    # Labels
    labels_map=load_labels(args.labels)
    labels_list=labels_to_list(labels_map) if labels_map else []

    keep0_names, keep0_ids0, keep0_ids1 = parse_keep(args.cam0_keep, VEHICLES)
    keep1_names, keep1_ids0, keep1_ids1 = parse_keep(args.cam1_keep, {"person"}|ANIMALS)

    # Intérpretes
    t0=args.threads0 if args.threads0 is not None else args.threads
    t1=args.threads1 if args.threads1 is not None else args.threads
    intrp0=Interpreter(model_path=args.model, num_threads=max(1,t0)); intrp0.allocate_tensors()
    intrp1=Interpreter(model_path=args.model, num_threads=max(1,t1)); intrp1.allocate_tensors()
    in0=intrp0.get_input_details()[0]; out0=intrp0.get_output_details(); ih0,iw0=int(in0["shape"][1]),int(in0["shape"][2])
    in1=intrp1.get_input_details()[0]; out1=intrp1.get_output_details(); ih1,iw1=int(in1["shape"][1]),int(in1["shape"][2])

    # Captura
    cap0=open_capture(args.cam0, args.cap_w, args.cap_h, args.fourcc)
    cap1=open_capture(args.cam1, args.cap_w, args.cap_h, args.fourcc)
    grab0=LatestFrame(cap0); grab1=LatestFrame(cap1)

    # FSM con margen configurable
    fsm=TrafficLightFSM(hold_sec=args.hold_sec)

    cv2.namedWindow("2cam", cv2.WINDOW_NORMAL)
    fcount0=fcount1=0
    last_dets0=[]; last_dets1=[]

    try:
        while True:
            ok0,f0=grab0.read()
            ok1,f1=grab1.read()
            if not ok0 and not ok1:
                time.sleep(0.005); continue

            persons_total=animals_total=vehicles_total=0

            # ---- CAM0 (vehículos) ----
            if ok0:
                H0,W0=f0.shape[:2]
                y0a=max(0,min(H0,int(args.crop0_top*H0))); y0b=max(0,min(H0,int(args.crop0_bottom*H0)))
                if y0b<=y0a: y0a,y0b=0,H0
                roi0=f0[y0a:y0b,:,:]; H0r=roi0.shape[0]
                area_min0=(args.mar0 if args.mar0 is not None else args.min_area_ratio)*W0*H0
                fcount0+=1; run0=(fcount0 % max(1,args.infer_every) == 0)
                dets0=last_dets0
                if run0:
                    rgb0=cv2.cvtColor(roi0, cv2.COLOR_BGR2RGB)
                    resized0=cv2.resize(rgb0,(iw0,ih0))
                    inp0=quantize_if_needed(resized0,in0["dtype"],in0.get("quantization"))
                    intrp0.set_tensor(in0["index"], np.expand_dims(inp0,0))
                    intrp0.invoke()
                    outs0=[intrp0.get_tensor(o['index']) for o in out0]
                    if len(outs0)>=4:
                        boxes0=squeeze01(outs0[0]); classes0=squeeze01(outs0[1]).astype(np.int32)
                        scores0=squeeze01(outs0[2]).astype(np.float32); num0=int(np.squeeze(outs0[3]))
                    elif len(outs0)==3:
                        boxes0=squeeze01(outs0[0]); classes0=squeeze01(outs0[1]).astype(np.int32)
                        scores0=squeeze01(outs0[2]).astype(np.float32); num0=min(len(scores0),len(classes0),len(boxes0))
                    else:
                        boxes0=classes0=scores0=[]; num0=0
                    thr0=args.thr0 if args.thr0 is not None else args.threshold
                    cand_boxes0=[]; cand_scores0=[]; cand_disp0=[]
                    for i in range(num0):
                        s=float(scores0[i])
                        if s<thr0: continue
                        cls_id=int(classes0[i]); name,_=resolve_name(cls_id, labels_list)
                        if not allowed(name, cls_id, keep0_names, keep0_ids0, keep0_ids1): continue
                        y1,x1,y2,x2=boxes0[i]
                        x1p=int(x1*W0); x2p=int(x2*W0); y1p=int(y1*H0r)+y0a; y2p=int(y2*H0r)+y0a
                        if (x2p-x1p)*(y2p-y1p) < area_min0: continue
                        cand_boxes0.append([x1p,y1p,x2p,y2p]); cand_scores0.append(s); cand_disp0.append((name,s,cls_id))
                    if args.topk0 and len(cand_scores0)>args.topk0:
                        idx=np.argsort(np.array(cand_scores0))[-args.topk0:]
                        cand_boxes0=[cand_boxes0[i] for i in idx]; cand_scores0=[cand_scores0[i] for i in idx]; cand_disp0=[cand_disp0[i] for i in idx]
                    keep0=nms_xyxy(cand_boxes0,cand_scores0,args.nms_iou) if cand_boxes0 else []
                    dets0=[(cand_boxes0[k][0],cand_boxes0[k][1],cand_boxes0[k][2],cand_boxes0[k][3], cand_disp0[k][0],cand_disp0[k][1],cand_disp0[k][2]) for k in keep0]
                    last_dets0=dets0
                for (x1,y1,x2,y2,name,score,cls_id) in dets0:
                    if categorize(name)=="vehicle":
                        vehicles_total+=1
                        cv2.rectangle(f0,(x1,y1),(x2,y2),(0,255,0),2)

            # ---- CAM1 (personas/animales) ----
            if ok1:
                H1,W1=f1.shape[:2]
                y1a=max(0,min(H1,int(args.crop1_top*H1))); y1b=max(0,min(H1,int(args.crop1_bottom*H1)))
                if y1b<=y1a: y1a,y1b=0,H1
                roi1=f1[y1a:y1b,:,:]; H1r=roi1.shape[0]
                area_min1=(args.mar1 if args.mar1 is not None else args.min_area_ratio)*W1*H1
                fcount1+=1; run1=(fcount1 % max(1,args.infer_every) == 0)
                dets1=last_dets1
                if run1:
                    rgb1=cv2.cvtColor(roi1, cv2.COLOR_BGR2RGB)
                    resized1=cv2.resize(rgb1,(iw1,ih1))
                    inp1=quantize_if_needed(resized1,in1["dtype"],in1.get("quantization"))
                    intrp1.set_tensor(in1["index"], np.expand_dims(inp1,0))
                    intrp1.invoke()
                    outs1=[intrp1.get_tensor(o['index']) for o in out1]
                    if len(outs1)>=4:
                        boxes1=squeeze01(outs1[0]); classes1=squeeze01(outs1[1]).astype(np.int32)
                        scores1=squeeze01(outs1[2]).astype(np.float32); num1=int(np.squeeze(outs1[3]))
                    elif len(outs1)==3:
                        boxes1=squeeze01(outs1[0]); classes1=squeeze01(outs1[1]).astype(np.int32)
                        scores1=squeeze01(outs1[2]).astype(np.float32); num1=min(len(scores1),len(classes1),len(boxes1))
                    else:
                        boxes1=classes1=scores1=[]; num1=0
                    thr1=args.thr1 if args.thr1 is not None else args.threshold
                    cand_boxes1=[]; cand_scores1=[]; cand_disp1=[]
                    for i in range(num1):
                        s=float(scores1[i])
                        if s<thr1: continue
                        cls_id=int(classes1[i]); name,_=resolve_name(cls_id, labels_list)
                        if not allowed(name, cls_id, keep1_names, keep1_ids0, keep1_ids1): continue
                        y1,x1,y2,x2=boxes1[i]
                        x1p=int(x1*W1); x2p=int(x2*W1); y1p=int(y1*H1r)+y1a; y2p=int(y2*H1r)+y1a
                        if (x2p-x1p)*(y2p-y1p) < area_min1: continue
                        cand_boxes1.append([x1p,y1p,x2p,y2p]); cand_scores1.append(s); cand_disp1.append((name,s,cls_id))
                    if args.topk1 and len(cand_scores1)>args.topk1:
                        idx=np.argsort(np.array(cand_scores1))[-args.topk1:]
                        cand_boxes1=[cand_boxes1[i] for i in idx]; cand_scores1=[cand_scores1[i] for i in idx]; cand_disp1=[cand_disp1[i] for i in idx]
                    keep1=nms_xyxy(cand_boxes1,cand_scores1,args.nms_iou) if cand_boxes1 else []
                    dets1=[(cand_boxes1[k][0],cand_boxes1[k][1],cand_boxes1[k][2],cand_boxes1[k][3], cand_disp1[k][0],cand_disp1[k][1],cand_disp1[k][2]) for k in keep1]
                    last_dets1=dets1
                for (x1,y1,x2,y2,name,score,cls_id) in dets1:
                    cat=categorize(name)
                    if cat=="person":
                        persons_total+=1; color=(255,255,0)
                    elif cat=="animal":
                        animals_total+=1; color=(255,0,255)
                    else:
                        color=(255,255,255)
                    cv2.rectangle(f1,(x1,y1),(x2,y2),color,2)

            # ---- FSM (histeresis estable) ----
            color_txt, tleft = fsm.update(persons_total, animals_total, vehicles_total)

            # ---- Canvas final ----
            canvas=stack_frames(f0 if ok0 else None, f1 if ok1 else None, args.layout)
            if canvas is None:
                continue
            put_banner(canvas, persons_total, animals_total, vehicles_total, color_txt, tleft)

            cv2.imshow("2cam", canvas)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

    finally:
        try: grab0.close()
        except: pass
        try: grab1.close()
        except: pass
        try: cap0.release(); cap1.release()
        except: pass
        try: cv2.destroyAllWindows()
        except: pass

if __name__=="__main__":
    main()

