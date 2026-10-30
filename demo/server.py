"""板子 → PC 实时推理看板（零第三方 Web 框架，仅需 pyserial）。

数据链路：
    STM32F103 --USART1/CH340--> COM 口 --> 本脚本解析 --> HTTP 网页大屏

用法（仓库根目录）：
    python demo/server.py --port COM6          # 读真实板子
    python demo/server.py                      # 自动找 CH340 串口
    python demo/server.py --simulate           # 无板子时演示
    python demo/server.py --replay log.txt     # 回放已保存的串口日志

然后浏览器打开 http://127.0.0.1:8765

页面每 500 ms 拉一次 /api/state，显示：当前类别/置信度、11 类概率、
板上 DWT 实测时延（cycles / µs）、最近记录、板端信息。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 依赖引导：把仓库内 .deps 或环境变量 EDGE_DEPS 指向的目录加入 sys.path；
# 若都没有，安装 pyserial 即可（pip install pyserial）。
_ROOT0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in (os.path.join(_ROOT0, ".deps"), os.environ.get("EDGE_DEPS", "")):
    if _d and os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
META_PATH = os.path.join(ROOT, "models", "modrec_meta.json")

RE_SEP = re.compile(r"^=+$")
RE_MODEL = re.compile(r"model\s*:\s*(\S+)")
RE_INPUT = re.compile(r"input\s*:\s*C=(\d+)\s+L=(\d+)")
RE_CLOCK = re.compile(r"clock\s*:\s*(\d+)\s*Hz\s*\((\w+)\)")
RE_DEMO = re.compile(r"demo\s*:\s*true=(\d+),\s*pc_ref_pred=(\d+)")
RE_CLASS = re.compile(r"class\s+(\d+)\s*:\s*prob=([0-9.]+)")
RE_PRED = re.compile(r"pred\s*:\s*(\d+)\s*\(ref\s*(\d+)\)\s*(MATCH|MISMATCH)")
RE_LAT_CYC = re.compile(r"latency:\s*min=(\d+)\s+avg=(\d+)\s+max=(\d+)\s+cycles")
RE_LAT_US = re.compile(r"latency:\s*min=([0-9.]+)\s+avg=([0-9.]+)\s+max=([0-9.]+)\s*us")


def load_classes():
    try:
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        return list(meta["classes"])
    except Exception:
        return [f"class{i}" for i in range(11)]


class BoardState:
    """线程安全地保存最新一帧 + 最近历史。"""

    def __init__(self, classes):
        self.lock = threading.Lock()
        self.classes = classes
        self.connected = False
        self.source = "未连接"
        self.frame = {}
        self.history = []      # 最近 60 帧的摘要
        self.frame_count = 0
        self.started_at = time.time()

    def set_connection(self, connected, source):
        with self.lock:
            self.connected = connected
            self.source = source

    def push(self, frame):
        with self.lock:
            self.frame = frame
            self.frame_count += 1
            self.history.append({
                "t": round(time.time(), 3),
                "pred": frame.get("pred"),
                "true": frame.get("true"),
                "conf": frame.get("confidence"),
                "us": frame.get("lat_us", {}).get("avg"),
            })
            del self.history[:-60]

    def snapshot(self):
        with self.lock:
            data = dict(self.frame)
            data.update({
                "connected": self.connected,
                "source": self.source,
                "frame_count": self.frame_count,
                "uptime": round(time.time() - self.started_at, 1),
                "classes": self.classes,
                "history": list(self.history),
            })
            return data


def parse_line(line, acc):
    """把一行串口文本解析进 acc（当前帧字典），整帧结束返回 dict 或 None。"""
    line = line.strip()
    if not line:
        return None
    if RE_SEP.match(line):
        # 分隔线：若上一段已收集到 pred+latency，则视为一帧结束
        if "pred" in acc and "lat_us" in acc:
            frame = dict(acc)
            acc.clear()
            return frame
        acc.clear()   # 表头分隔线，丢弃上一段残缺内容
        return None

    m = RE_MODEL.search(line)
    if m and "model" not in acc:
        acc["model"] = m.group(1)
    m = RE_INPUT.search(line)
    if m:
        acc["input_c"], acc["input_l"] = int(m.group(1)), int(m.group(2))
    m = RE_CLOCK.search(line)
    if m:
        acc["clock_hz"], acc["clock_src"] = int(m.group(1)), m.group(2)
    m = RE_DEMO.search(line)
    if m:
        acc["true"], acc["ref_pred"] = int(m.group(1)), int(m.group(2))
    m = RE_CLASS.search(line)
    if m:
        acc.setdefault("probs", {})[int(m.group(1))] = float(m.group(2))
    m = RE_PRED.search(line)
    if m:
        acc["pred"], acc["ref_pred"], acc["match"] = (
            int(m.group(1)), int(m.group(2)), m.group(3) == "MATCH")
    m = RE_LAT_CYC.search(line)
    if m:
        acc["lat_cycles"] = {"min": int(m.group(1)), "avg": int(m.group(2)),
                             "max": int(m.group(3))}
    m = RE_LAT_US.search(line)
    if m:
        acc["lat_us"] = {"min": float(m.group(1)), "avg": float(m.group(2)),
                         "max": float(m.group(3))}
    return None


def frame_to_state(frame, classes):
    probs = frame.get("probs", {})
    prob_list = [probs.get(i, 0.0) for i in range(len(classes))]
    pred = frame.get("pred", 0)
    out = {
        "model": frame.get("model", "?"),
        "input": [frame.get("input_c"), frame.get("input_l")],
        "clock_hz": frame.get("clock_hz"),
        "clock_src": frame.get("clock_src"),
        "true": frame.get("true"),
        "ref_pred": frame.get("ref_pred"),
        "pred": pred,
        "pred_name": classes[pred] if 0 <= pred < len(classes) else "?",
        "confidence": prob_list[pred] if 0 <= pred < len(prob_list) else 0.0,
        "match": frame.get("match"),
        "probs": prob_list,
        "lat_cycles": frame.get("lat_cycles"),
        "lat_us": frame.get("lat_us"),
        "ts": round(time.time(), 3),
    }
    return out


class SerialReader(threading.Thread):
    """后台读串口，解析完整帧后推给 BoardState；断线自动重连。"""

    def __init__(self, state, port=None, baud=115200, replay=None):
        super().__init__(daemon=True)
        self.state = state
        self.port = port
        self.baud = baud
        self.replay = replay
        self.acc = {}
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _detect_port(self):
        try:
            from serial.tools import list_ports
            for p in list_ports.comports():
                text = f"{p.device} {p.description}"
                if "CH340" in text or "USB-SERIAL" in text or "USB Serial" in text:
                    return p.device
            ports = [p.device for p in list_ports.comports()]
            return ports[0] if ports else None
        except Exception:
            return None

    def _handle_line(self, raw):
        frame = parse_line(raw, self.acc)
        if frame:
            self.state.push(frame_to_state(frame, self.state.classes))

    def run(self):
        if self.replay:
            self.state.set_connection(True, f"回放 {os.path.basename(self.replay)}")
            with open(self.replay, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if self._stop.is_set():
                        return
                    self._handle_line(line)
                    time.sleep(0.02)
            self.state.set_connection(False, "回放结束")
            return

        try:
            import serial  # pyserial
        except ImportError:
            print("[serial] 缺少 pyserial，请先执行: pip install pyserial", flush=True)
            self.state.set_connection(False, "缺少 pyserial（pip install pyserial）")
            return

        while not self._stop.is_set():
            port = self.port or self._detect_port()
            if not port:
                self.state.set_connection(False, "未找到串口（插好板子后自动重试）")
                time.sleep(3)
                continue
            try:
                ser = serial.Serial(port, self.baud, timeout=1)
                # CH340 的 DTR 常接板子 RST：pyserial 默认拉高会把板子按在复位里，
                # 必须立刻释放；rts 同理。
                try:
                    ser.dtr = False
                    ser.rts = False
                except Exception:
                    pass
                time.sleep(0.3)
                self.state.set_connection(True, port)
                print(f"[serial] connected {port} @ {self.baud}", flush=True)
                buf = b""
                while not self._stop.is_set():
                    chunk = ser.read(256)
                    if not chunk:
                        continue
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        self._handle_line(line.decode("utf-8", errors="replace"))
                ser.close()
            except Exception as exc:
                print(f"[serial] {port}: {exc}", flush=True)
                self.state.set_connection(False, f"{port} 连接失败，重试中")
                time.sleep(3)


class Simulator(threading.Thread):
    """无板子时的演示数据（概率之和为 1，时延按实测 ~33 ms）。"""

    def __init__(self, state):
        super().__init__(daemon=True)
        self.state = state
        self._stop = threading.Event()
        self.n = 0

    def stop(self):
        self._stop.set()

    def run(self):
        self.state.set_connection(True, "模拟模式（未连接板子）")
        rnd = random.Random(7)
        classes = self.state.classes
        order = ["AM-DSB", "AM-SSB", "WBFM", "QPSK", "QAM16", "QAM64",
                 "8PSK", "BPSK", "CPFSK", "GFSK", "PAM4"]
        while not self._stop.is_set():
            self.n += 1
            pred = self.n // 6 % len(classes)
            base = [rnd.uniform(0.01, 0.06) for _ in range(len(classes))]
            base[pred] = rnd.uniform(0.55, 0.85)
            s = sum(base)
            probs = [round(x / s, 4) for x in base]
            probs[pred] = round(1 - sum(p for i, p in enumerate(probs) if i != pred), 4)
            frame = {
                "model": "radio_ml_cnn", "input": [2, 126],
                "clock_hz": 72000000, "clock_src": "PLL", "true": pred,
                "ref_pred": pred, "pred": pred,
                "pred_name": classes[pred], "confidence": probs[pred],
                "match": True, "probs": probs,
                "lat_cycles": {"min": 2400000, "avg": 2400411, "max": 2400420},
                "lat_us": {"min": 33330.0, "avg": 33339.04, "max": 33340.0},
                "ts": round(time.time(), 3),
            }
            self.state.push(frame)
            time.sleep(1.2)


class Handler(BaseHTTPRequestHandler):
    server_version = "EdgeInferenceDemo/1.0"

    def log_message(self, fmt, *args):
        pass  # 安静一点

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            with open(os.path.join(STATIC_DIR, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._send(200, json.dumps(self.server.board.snapshot(),
                                       ensure_ascii=False))
        elif path == "/api/health":
            self._send(200, json.dumps({"ok": True}))
        else:
            self._send(404, json.dumps({"error": "not found"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="串口，如 COM6；默认自动找 CH340")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--http-port", type=int, default=8765)
    ap.add_argument("--simulate", action="store_true", help="无板子演示模式")
    ap.add_argument("--replay", default=None, help="回放串口日志文件")
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    classes = load_classes()
    state = BoardState(classes)

    if args.simulate:
        reader = Simulator(state)
    else:
        reader = SerialReader(state, port=args.port, baud=args.baud,
                              replay=args.replay)
    reader.start()

    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", args.http_port), Handler)
    except OSError as exc:
        print(f"[http] 端口 {args.http_port} 已被占用（{exc}），可能已经开过看板，"
              f"直接打开浏览器即可。")
        if not args.no_browser:
            webbrowser.open(f"http://127.0.0.1:{args.http_port}")
        return
    httpd.board = state
    print(f"[http] 看板地址: http://127.0.0.1:{args.http_port}")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(
            f"http://127.0.0.1:{args.http_port}")).start()
    print(f"[info] 模式: {'模拟' if args.simulate else ('回放' if args.replay else '串口')}"
          f" | 类别数: {len(classes)}")
    print("[info] Ctrl+C 退出")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        httpd.server_close()


if __name__ == "__main__":
    main()





