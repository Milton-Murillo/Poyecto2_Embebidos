#!/usr/bin/env python3
# coco.py — Vehículos + animales seleccionados (cat, dog, horse, cow, sheep)
import argparse, time, os, sys
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

# ---------- FILTRO: Vehículos + animales seleccionados ----------
VEHICLES = {"bicycle","car","motorcycle","bus","train","truck"}
SELECTED_ANIMALS = {"cat","dog","horse","cow","sheep","person"}

KEEP_NAMES_DEFAULT = set(VEHICLES | SELECTED_ANIMALS)

# Deshabilitamos filtro por IDs para evitar descalces 0/1-based (filtramos por nombre)
IDS_KEEP_0 = set()
IDS_KEEP_1 = set()

# ---------- Utils ----------
def load_labels(path: str|None):
    if not path or not os.path.exists(path):
        return None
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
    # intenta 0-based y 1-based con labels_list
    for off in (0, -1):
        idx = cls_id + off
        if 0 <= idx < len(labels_list):
            name = labels_list[idx]
            if name != "???":
                return name, off
    # fallback a COCO80
    for off in (0, -1):
        idx = cls_id + off
        if 0 <= idx < len(COCO80):
            return COCO80[idx], off
    return None, None

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

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ruta al .tflite (SSD)")
    ap.add_argument("--labels", default=None, help="Ruta a labels.txt / labelmap.txt")
    ap.add_argument("--source", default="0", help="Índice /dev/videoN o ruta de video")
    ap.add_argument("--cap_w", type=int, default=1280)
    ap.add_argument("--cap_h", type=int, default=720)
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--nms_iou", type=float, default=0.50, help="IoU para NMS (0.3–0.6 sugerido)")
    ap.add_argument("--min_area_ratio", type=float, default=0.003, help="Área mínima relativa")
    ap.add_argument("--show_fps", action="store_true", default=True)
    ap.add_argument("--debug", action="store_true", help="Imprime detecciones crudas antes del filtro")
    args = ap.parse_args()

    # Candado anti-doble instancia por cámara (evita múltiples ventanas si ejecutas dos veces)
    src_key = args.source if not args.source.isdigit() else f"/dev/video{args.source}"
    lock_path = f"/tmp/coco_py_lock_{os.path.basename(str(src_key))}"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(lock_fd, str(os.getpid()).encode("ascii"))
        os.close(lock_fd)
    except FileExistsError:
        print(f"[ERROR] Ya hay otra instancia usando {src_key}. Cierra la anterior o borra {lock_path}.", file=sys.stderr)
        sys.exit(1)

    labels_map = load_labels(args.labels)
    labels_list = labels_to_list(labels_map) if labels_map else []

    # Intérprete y tamaño de entrada
    interpreter = Interpreter(model_path=args.model, num_threads=max(1, args.threads))
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()
    ih, iw = int(in_det["shape"][1]), int(in_det["shape"][2])

    # Cámara
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2 if isinstance(src,int) else 0)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cap_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cap_h)
        cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("No pude abrir la fuente de video.", file=sys.stderr)
        try: os.unlink(lock_path)
        except Exception: pass
        sys.exit(1)

    # Ventana única y fija
    WIN = "COCO"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    print("FILTRO activo:", sorted(list(KEEP_NAMES_DEFAULT)))
    prev=time.time(); fps=0.0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok: break
            H,W = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Entrada exacta (iw, ih)
            resized = cv2.resize(frame_rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)
            inp = quantize_if_needed(resized, in_det["dtype"], in_det.get("quantization"))
            interpreter.set_tensor(in_det["index"], np.expand_dims(inp,0))
            interpreter.invoke()

            # Salidas SSD (3 o 4 tensores)
            outs = [interpreter.get_tensor(o['index']) for o in out_det]
            if len(outs) >= 4:
                boxes_n  = squeeze01(outs[0])
                classes_v= squeeze01(outs[1]).astype(np.int32)
                scores_v = squeeze01(outs[2]).astype(np.float32)
                num      = int(np.squeeze(outs[3]))
            elif len(outs) == 3:
                boxes_n  = squeeze01(outs[0])
                classes_v= squeeze01(outs[1]).astype(np.int32)
                scores_v = squeeze01(outs[2]).astype(np.float32)
                num      = min(len(scores_v), len(classes_v), len(boxes_n))
            else:
                print(f"El modelo no parece SSD estándar (salidas={len(outs)}).", file=sys.stderr)
                break

            cand_boxes, cand_scores, cand_disp = [], [], []
            area_min = args.min_area_ratio * W * H

            for i in range(num):
                s = float(scores_v[i])
                if s < args.threshold:
                    continue
                cls_id = int(classes_v[i])
                name, _off = resolve_name(cls_id, labels_list)

                if args.debug:
                    print(f"[RAW] id={cls_id} name={name} score={s:.2f}")

                # SOLO nombres definidos en KEEP_NAMES_DEFAULT
                name_ok = (name is not None) and (name.lower() in KEEP_NAMES_DEFAULT)
                if not name_ok:
                    continue

                # [ymin,xmin,ymax,xmax] -> pixeles
                y1,x1,y2,x2 = boxes_n[i]
                x1p, y1p = int(x1*W), int(y1*H)
                x2p, y2p = int(x2*W), int(y2*H)
                if (x2p-x1p)*(y2p-y1p) < area_min:
                    continue

                cand_boxes.append([x1p,y1p,x2p,y2p])
                cand_disp.append(f"{name} {s:.2f}")
                cand_scores.append(s)

            keep_idx = nms_xyxy(cand_boxes, cand_scores, args.nms_iou) if cand_boxes else []

            kept = 0
            for k in keep_idx:
                (x1,y1,x2,y2) = cand_boxes[k]
                cv2.rectangle(frame_bgr,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame_bgr, cand_disp[k], (x1, max(0,y1-7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,255,0), 2, cv2.LINE_AA)
                kept += 1

            if args.show_fps:
                now=time.time(); fps=0.9*fps+0.1*(1.0/(now-prev)); prev=now
                cv2.putText(frame_bgr,f"FPS {fps:.1f}",(8,20), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),3,cv2.LINE_AA)
                cv2.putText(frame_bgr,f"FPS {fps:.1f}",(8,20), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(frame_bgr,f"kept:{kept}",(8,40),   cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),3,cv2.LINE_AA)
                cv2.putText(frame_bgr,f"kept:{kept}",(8,40),   cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1,cv2.LINE_AA)

            cv2.imshow(WIN, frame_bgr)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try: os.unlink(lock_path)
        except Exception: pass

if __name__=="__main__":
    main()

