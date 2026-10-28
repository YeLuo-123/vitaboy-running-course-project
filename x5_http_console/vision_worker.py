#!/usr/bin/env python3
"""Loopback-only YOLO-World v2 + CLIP worker, isolated from the ROS executor."""
import argparse
import json
import os
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import cv2
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from object_prompts import resolve_prompts

ROOT = Path(__file__).resolve().parent


def decode_detections(output, prompt, scale, padding, shape, threshold=.20):
    """Export already applies sigmoid. Boxes are input-image xywh in pixels."""
    rows = np.asarray(output[0], dtype=np.float32).T
    if rows.ndim != 2 or rows.shape[1] != 4+len(prompt['texts']):
        raise RuntimeError('YOLO-World 输出形状不符')
    scores = rows[:, 4:]
    if not np.isfinite(rows).all() or scores.min() < -.001 or scores.max() > 1.001:
        raise RuntimeError('YOLO-World 输出数值无效')
    classes = scores.argmax(axis=1)
    confidence = scores.max(axis=1)
    selected = np.where((classes < prompt['positive_count']) & (confidence >= threshold))[0]
    boxes, values, labels = [], [], []
    height, width = shape[:2]
    for index in selected:
        x, y, w, h = rows[index, :4]
        x1 = max(0., min(float(width), (x-w/2-padding[0])/scale))
        y1 = max(0., min(float(height), (y-h/2-padding[1])/scale))
        x2 = max(0., min(float(width), (x+w/2-padding[0])/scale))
        y2 = max(0., min(float(height), (y+h/2-padding[1])/scale))
        if x2-x1 < 6 or y2-y1 < 6:
            continue
        boxes.append([x1, y1, x2-x1, y2-y1]); values.append(float(confidence[index]))
        labels.append(prompt['texts'][int(classes[index])])
    keep = cv2.dnn.NMSBoxes(boxes, values, threshold, .45) if boxes else []
    detections = []
    for i in np.asarray(keep).reshape(-1):
        x, y, w, h = boxes[int(i)]
        detections.append({'label': prompt['name'], 'prompt': labels[int(i)],
                           'score': round(values[int(i)], 4),
                           'box': [round(float(x), 1), round(float(y), 1), round(float(x+w), 1), round(float(y+h), 1)]})
    return detections


class Camera:
    def __init__(self, device):
        self.device = device; self.lock = threading.Lock(); self.frame = None
        self.stamp = 0.; self.sequence = 0; self.error = '等待相机图像'; self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        cv2.setNumThreads(1)
        while self.running:
            capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            try:
                if not capture.isOpened():
                    raise RuntimeError('无法打开相机 '+self.device)
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                capture.set(cv2.CAP_PROP_FPS, 10)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                while self.running:
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError('相机图像读取失败')
                    with self.lock:
                        self.frame = frame; self.stamp = time.monotonic()
                        self.sequence += 1; self.error = None
                    time.sleep(.03)
            except Exception as exc:
                with self.lock: self.error = str(exc)
            finally:
                capture.release()
            time.sleep(1.)

    def snapshot(self):
        with self.lock:
            if self.frame is None or time.monotonic()-self.stamp > 2 or self.error:
                raise RuntimeError(self.error or '相机图像已过期')
            return self.frame.copy(), self.sequence, self.stamp

    def status(self):
        with self.lock:
            return {'camera_ready': self.frame is not None and time.monotonic()-self.stamp < 2 and not self.error,
                    'camera_error': self.error, 'camera_sequence': self.sequence}


class Engine:
    def __init__(self, models, camera, image_size=640, threads=2):
        self.camera = camera; self.size = image_size; self.lock = threading.Lock()
        self.started = time.monotonic(); self.last_result = None; self.last_jpeg = None
        self.job_id = ''; self.prompt = None; self.features = None; self.error = None
        options = ort.SessionOptions(); options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.detector = ort.InferenceSession(str(models/'yolov8s-worldv2.onnx'), options,
                                             providers=['CPUExecutionProvider'])
        self.text_encoder = ort.InferenceSession(str(models/'clip_text_quantized.onnx'), options,
                                                 providers=['CPUExecutionProvider'])
        self.tokenizer = Tokenizer.from_file(str(models/'clip_tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=77)
        self.tokenizer.enable_padding(length=77, pad_id=49407, pad_token='<|endoftext|>')

    def configure(self, data):
        prompt = resolve_prompts(data.get('item'), data.get('english_hint', ''))
        job_id = data.get('job_id')
        if not isinstance(job_id, str) or not 1 <= len(job_id) <= 80:
            raise ValueError('任务编号无效')
        with self.lock:
            tokens = self.tokenizer.encode_batch(prompt['texts'])
            ids = np.asarray([t.ids for t in tokens], dtype=np.int64)
            features = self.text_encoder.run(['text_embeds'], {'input_ids': ids})[0]
            features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
            self.features = features[np.newaxis].astype(np.float32)
            self.prompt = prompt; self.job_id = job_id; self.last_result = None; self.last_jpeg = None; self.error = None
        return {'job_id': job_id, 'prompt': prompt, 'ready': True}

    def detect(self, data):
        with self.lock:
            if data.get('job_id') != self.job_id or self.prompt is None:
                raise ValueError('视觉任务已失效，请重新开始')
            frame, sequence, captured = self.camera.snapshot()
            if sequence <= int(data.get('after_sequence', -1)):
                raise RuntimeError('没有新的相机图像')
            threshold = float(data.get('threshold', .20))
            if not .10 <= threshold <= .90:
                raise ValueError('识别阈值无效')
            height, width = frame.shape[:2]; scale = min(self.size/height, self.size/width)
            new_w, new_h = int(round(width*scale)), int(round(height*scale))
            pad_x, pad_y = (self.size-new_w)//2, (self.size-new_h)//2
            image = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
            image[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = cv2.resize(frame, (new_w, new_h))
            image = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)/255.
            before = time.monotonic()
            output = self.detector.run(None, {'images': image, 'txt_feats': self.features})[0]
            elapsed = time.monotonic()-before
            detections = decode_detections(output, self.prompt, scale, (pad_x, pad_y), frame.shape, threshold)
            result = {'job_id': self.job_id, 'sequence': sequence, 'captured_monotonic': captured,
                      'capture_age_sec': round(before-captured, 3), 'inference_sec': round(elapsed, 3),
                      'width': width, 'height': height, 'detections': detections,
                      'model': 'YOLO-World v2 Small', 'image_size': self.size}
            for found in detections:
                x1, y1, x2, y2 = (int(v) for v in found['box'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 80), 2)
                cv2.putText(frame, f"{found['prompt']} {found['score']:.2f}", (x1, max(18, y1-7)),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (40, 220, 80), 2)
            ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self.last_jpeg = jpeg.tobytes() if ok else None; self.last_result = result
            return result

    def status(self):
        return {'ready': True, 'model': 'YOLO-World v2 Small', 'backend': 'ONNX Runtime CPU / 2 threads',
                'image_size': self.size, **self.camera.status(), 'error': self.error,
                'last_result': self.last_result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8091)
    parser.add_argument('--camera', default='/dev/video0')
    parser.add_argument('--image-size', type=int, default=640, choices=(320, 416, 640))
    parser.add_argument('--models', type=Path, default=ROOT/'models')
    args = parser.parse_args()
    camera = Camera(args.camera)
    engine = Engine(args.models, camera, args.image_size)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, body, kind='application/json; charset=utf-8'):
            raw = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code); self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(raw))); self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try: self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError): pass

        def do_GET(self):
            if self.path == '/health': return self.reply(200, engine.status())
            if self.path == '/frame':
                jpeg = engine.last_jpeg
                if jpeg is None:
                    try:
                        frame, _, _ = camera.snapshot(); ok, data = cv2.imencode('.jpg', frame)
                        if not ok: raise RuntimeError('相机预览编码失败')
                        jpeg = data.tobytes()
                    except Exception as exc: return self.reply(503, {'error': str(exc)})
                return self.reply(200, jpeg, 'image/jpeg')
            self.reply(404, {'error': '接口不存在'})

        def do_POST(self):
            try:
                count = int(self.headers.get('Content-Length', '0'))
                if not 0 < count <= 4096: raise ValueError('请求大小无效')
                data = json.loads(self.rfile.read(count))
                if not isinstance(data, dict): raise ValueError('请求需要 JSON 对象')
                if self.path == '/configure': result = engine.configure(data)
                elif self.path == '/detect': result = engine.detect(data)
                else: return self.reply(404, {'error': '接口不存在'})
                self.reply(200, result)
            except Exception as exc:
                self.reply(400 if isinstance(exc, ValueError) else 503, {'error': str(exc)})

        def log_message(self, *_): pass

    print(json.dumps(engine.status(), ensure_ascii=False), flush=True)
    try: ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
    finally: camera.running = False


if __name__ == '__main__': main()
