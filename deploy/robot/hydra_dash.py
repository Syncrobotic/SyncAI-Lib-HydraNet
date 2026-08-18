#!/usr/bin/env python3
"""
HydraNet Playground - Lite3 hardware dashboard
Serves a web page showing live hardware telemetry from the DEEP Robotics Lite3
(RK3588 motion host) plus a control panel scaffold.

Data sources (all read-only / passive except explicit control):
  - System HW: /proc, /sys
  - IMU + 12 joints: passive UDP listen on 43897 (jy_exe telemetry, code 0x0906)
  - Camera: robot's existing MediaMTX stream (WebRTC/HLS, path 'test')
  - Control: UDP to jy_exe 43893 (EthCommand). Only ESTOP-safe-stop is armed.

Pure Python 3 stdlib. No external deps.
"""

import contextlib
import glob
import http.server
import json
import os
import socket
import socketserver
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request

PORT = 8080
TELEM_PORT = 43897  # jy_exe sends RobotData here (we passively listen)
CMD_IP, CMD_PORT = "127.0.0.1", 43893
ROBOT_IP = "192.168.1.120"  # for camera stream URLs from the browser's perspective

# ---------------------------------------------------------------- robot telemetry
JOINT_NAMES = [
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "HL_hip",
    "HL_thigh",
    "HL_calf",
    "HR_hip",
    "HR_thigh",
    "HR_calf",
]

_robot = {
    "online": False,
    "tick": 0,
    "age_ms": None,
    "hz": 0.0,
    "imu": None,
    "joints": None,
    "last": 0.0,
}
_robot_lock = threading.Lock()


def _write_state(st):
    try:
        os.makedirs("/dev/shm/hydra", exist_ok=True)
        tmp = "/dev/shm/hydra/.robot_state.json"
        with open(tmp, "w") as fh:
            json.dump(st, fh)
        os.replace(tmp, "/dev/shm/hydra/robot_state.json")
    except Exception:
        pass


def telem_listener():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    with contextlib.suppress(Exception):
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    try:
        s.bind(("0.0.0.0", TELEM_PORT))
    except Exception as e:
        print("telem bind failed:", e)
        return
    s.settimeout(2.0)
    cnt = 0
    rate_t = time.time()
    hz = 0.0
    last_state_write = 0.0
    while True:
        try:
            d, _ = s.recvfrom(2048)
        except TimeoutError:
            with _robot_lock:
                if time.time() - _robot["last"] > 2.0:
                    _robot["online"] = False
            continue
        if len(d) < 12:
            continue
        code = struct.unpack_from("<I", d, 0)[0]
        now = time.time()
        if code == 0x0906 and len(d) == 380:
            tick = struct.unpack_from("<I", d, 12)[0]
            imu = struct.unpack_from("<9f", d, 20)
            joints = [struct.unpack_from("<4f", d, 56 + 16 * i) for i in range(12)]
            cnt += 1
            if now - rate_t >= 0.5:
                hz = cnt / (now - rate_t)
                cnt = 0
                rate_t = now
            with _robot_lock:
                _robot.update(
                    online=True,
                    tick=tick,
                    last=now,
                    hz=round(hz, 1),
                    imu={
                        "roll": imu[0],
                        "pitch": imu[1],
                        "yaw": imu[2],
                        "wroll": imu[3],
                        "wpitch": imu[4],
                        "wyaw": imu[5],
                        "ax": imu[6],
                        "ay": imu[7],
                        "az": imu[8],
                    },
                    joints=[
                        {
                            "name": JOINT_NAMES[i],
                            "pos": joints[i][0],
                            "vel": joints[i][1],
                            "tau": joints[i][2],
                            "temp": joints[i][3],
                        }
                        for i in range(12)
                    ],
                )
            # publish body attitude (deg) for hydra_infer's per-frame BEV ground plane
            if now - last_state_write >= 0.1:
                last_state_write = now
                _write_state({"roll": imu[0], "pitch": imu[1], "yaw": imu[2], "ts": now})
        elif code == 0x0901 and len(d) >= 212:
            # RobotStateUpload (official 0x0901 @50Hz): battery, ultrasound, odometry
            p = 12
            basic = struct.unpack_from("<i", d, p)[0]
            pos_world = struct.unpack_from("<3d", d, p + 80)
            vel_body = struct.unpack_from("<3d", d, p + 128)
            batt = struct.unpack_from("<d", d, p + 168)[0]
            ultra = struct.unpack_from("<2d", d, p + 184)
            with _robot_lock:
                _robot.update(
                    online=True,
                    last=now,
                    battery=round(batt, 1),
                    basic_state=basic,
                    ultrasound=[round(ultra[0], 3), round(ultra[1], 3)],
                    pos_world=[round(x, 3) for x in pos_world],
                    vel_body=[round(x, 3) for x in vel_body],
                )


def robot_snapshot():
    with _robot_lock:
        r = dict(_robot)
    if r.get("last"):
        r["age_ms"] = int((time.time() - r["last"]) * 1000)
    r.pop("last", None)
    return r


# ---------------------------------------------------------------- system sampler
_sys = {}
_sys_lock = threading.Lock()


def _read(p, default=""):
    try:
        with open(p) as f:
            return f.read().strip()
    except Exception:
        return default


def _cpu_times():
    out = {}
    for line in _read("/proc/stat").splitlines():
        if line.startswith("cpu") and line[3:4].isdigit():
            parts = line.split()
            vals = list(map(int, parts[1:]))
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            out[parts[0]] = (sum(vals), idle)
    return out


def sys_sampler():
    prev = _cpu_times()
    while True:
        time.sleep(1.0)
        cur = _cpu_times()
        usage = {}
        for k in cur:
            if k in prev:
                dt = cur[k][0] - prev[k][0]
                di = cur[k][1] - prev[k][1]
                usage[k] = round(100.0 * (dt - di) / dt, 1) if dt > 0 else 0.0
        prev = cur
        # per-core freq
        cores = []
        for i in range(len(usage)):
            f = _read("/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq" % i)
            mhz = int(f) // 1000 if f.isdigit() else None
            cores.append({"core": i, "use": usage.get("cpu%d" % i, 0.0), "mhz": mhz})
        # thermals
        temps = []
        for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
            t = _read(z + "/temp")
            ty = _read(z + "/type")
            if t.lstrip("-").isdigit():
                temps.append(
                    {"name": ty or os.path.basename(z), "c": round(int(t) / 1000.0, 1)}
                )
        # mem
        mem = {}
        for line in _read("/proc/meminfo").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                mem[k] = int(v.strip().split()[0])
        mem_total = mem.get("MemTotal", 0)
        mem_avail = mem.get("MemAvailable", 0)
        # disk
        try:
            st = os.statvfs("/")
            disk_total = st.f_blocks * st.f_frsize
            disk_free = st.f_bavail * st.f_frsize
        except Exception:
            disk_total = disk_free = 0
        # load / uptime
        load = _read("/proc/loadavg").split()[:3]
        up = float(_read("/proc/uptime", "0").split()[0] or 0)
        # wifi signal from /proc/net/wireless
        for line in _read("/proc/net/wireless").splitlines():
            if ":" in line and not line.strip().startswith("Inter"):
                p = line.split()
                with contextlib.suppress(Exception):
                    {
                        "iface": p[0].strip(":"),
                        "link": float(p[2].rstrip(".")),
                        "level_dbm": float(p[3].rstrip(".")),
                    }
        with _sys_lock:
            _sys.update(
                cores=cores,
                temps=temps,
                mem={
                    "total": mem_total,
                    "avail": mem_avail,
                    "used_pct": round(100 * (mem_total - mem_avail) / mem_total, 1)
                    if mem_total
                    else 0,
                },
                disk={
                    "total": disk_total,
                    "free": disk_free,
                    "used_pct": round(100 * (disk_total - disk_free) / disk_total, 1)
                    if disk_total
                    else 0,
                },
                load=load,
                uptime_s=int(up),
            )


def net_info():
    try:
        out = subprocess.run(
            ["ip", "-br", "-4", "addr"], capture_output=True, text=True, timeout=2
        ).stdout
        nets = []
        for line in out.splitlines():
            p = line.split()
            if len(p) >= 3 and p[0] not in ("lo",):
                nets.append({"iface": p[0], "state": p[1], "addr": p[2]})
        return nets
    except Exception:
        return []


_static = {}


def static_info():
    if _static:
        return _static
    _static.update(
        model=_read("/proc/device-tree/model").replace("\x00", "") or "unknown",
        kernel=_read("/proc/sys/kernel/osrelease"),
        hostname=_read("/proc/sys/kernel/hostname"),
        ncpu=len(
            [
                l
                for l in _read("/proc/stat").splitlines()
                if l.startswith("cpu") and l[3:4].isdigit()
            ]
        ),
    )
    return _static


def system_snapshot():
    with _sys_lock:
        s = dict(_sys)
    s["static"] = static_info()
    s["net"] = net_info()
    return s


# ---------------------------------------------------------------- control (teleop)
# Full command set from the beta comm spec V1.0.7. Movement streams axis commands while
# armed; the robot's own 250 ms axis-timeout plus a UI-heartbeat deadman keep it safe.
def _send_eth(code, value=0, typ=0):
    pkt = struct.pack("<III", code, value & 0xFFFFFFFF, typ)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(pkt, (CMD_IP, CMD_PORT))
    s.close()


def _send_vel(v):
    pkt = struct.pack("<III", 0x141, 8, 1) + struct.pack("<d", float(v))
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(pkt, (CMD_IP, CMD_PORT))
    s.close()


def _burst(code, value=0, typ=0, n=6, interval=0.02):
    """Send an EthCommand a few times to jy_exe:43893, mirroring how the real
    controller delivers a button press. Codes captured from the physical controller."""
    pkt = struct.pack("<III", code, value & 0xFFFFFFFF, typ)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for _ in range(n):
            s.sendto(pkt, (CMD_IP, CMD_PORT))
            time.sleep(interval)
    finally:
        s.close()


# Full teleop command set (beta comm spec V1.0.7). Discrete commands go through _burst;
# continuous movement is streamed by move_streamer() at 50 Hz while armed.
SIMPLE_CMDS = {
    "stand_toggle": 0x21010202,  # 起立/趴下 toggle
    "zero": 0x21010C05,  # 回零 (init joints)
    "mode_move": 0x21010D06,  # 移动模式
    "mode_inplace": 0x21010D05,  # 原地模式
    "ctrl_manual": 0x21010C02,  # 手动模式 (responds to handheld/this host)
    "ctrl_auto": 0x21010C03,  # 自主模式 (responds to perception host)
    "gait_low": 0x21010300,
    "gait_mid": 0x21010307,
    "gait_high": 0x21010303,
    "gait_grip": 0x21010402,  # 抓地越障
    "gait_highstep": 0x21010407,  # 高踏步越障
    "act_greet": 0x21010507,  # 打招呼
    "act_twist": 0x21010204,  # 扭身体
}
AXIS_FWD, AXIS_LAT, AXIS_YAW = 0x21010130, 0x21010131, 0x21010135
# conservative fractions of the +/-32767 axis range -- deliberately slow to start
FWD_MAX, LAT_MAX, YAW_MAX = 9000, 7000, 8000
_move = {"vx": 0.0, "vy": 0.0, "vyaw": 0.0, "armed": False, "last_hb": 0.0}
_move_lock = threading.Lock()


def _send_axis(sock, code, norm, mx):
    v = int(max(-1.0, min(1.0, norm)) * mx)
    sock.sendto(struct.pack("<III", code, v & 0xFFFFFFFF, 0), (CMD_IP, CMD_PORT))


def move_streamer():
    """Stream the three axis commands at 50 Hz while armed. The robot auto-stops after
    250 ms with no axis command, so this loop is the deadman: a UI heartbeat older than
    0.4 s zeroes velocity, and disarming stops the stream."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        with _move_lock:
            m = dict(_move)
        if m["armed"]:
            active = (time.time() - m["last_hb"]) < 0.4
            _send_axis(sock, AXIS_FWD, m["vx"] if active else 0.0, FWD_MAX)
            _send_axis(sock, AXIS_LAT, m["vy"] if active else 0.0, LAT_MAX)
            _send_axis(sock, AXIS_YAW, m["vyaw"] if active else 0.0, YAW_MAX)
        time.sleep(0.02)


def do_move(vx, vy, vyaw):
    with _move_lock:
        armed = _move["armed"]
        if armed:
            _move.update(vx=vx, vy=vy, vyaw=vyaw, last_hb=time.time())
    return {"ok": True, "armed": armed}


# Real command codes, captured from the physical controller and confirmed by the operator:
#   0x21010202  stand-up / crouch-down TOGGLE
#   0x21020C0E         soft emergency STOP (official spec)
def do_control(action):
    try:
        if action == "estop":
            with _move_lock:
                _move["armed"] = False  # e-stop always disarms teleop
            _burst(0x21020C0E, 0, 0, n=8)
            return {"ok": True, "msg": "軟急停已送出 (0x21020C0E)，操控已停用"}
        if action == "arm":
            _burst(0x21010D06, 0, 0, n=3)  # movement mode
            _burst(0x21010C02, 0, 0, n=3)  # manual mode
            with _move_lock:
                _move.update(armed=True, vx=0.0, vy=0.0, vyaw=0.0, last_hb=time.time())
            return {"ok": True, "armed": True, "msg": "操控已啟用（移動＋手動模式）"}
        if action == "disarm":
            with _move_lock:
                _move.update(armed=False, vx=0.0, vy=0.0, vyaw=0.0)
            return {"ok": True, "armed": False, "msg": "操控已停用"}
        if action in ("toggle", "stand", "sit"):
            _burst(0x21010202, 0, 0)
            return {"ok": True, "msg": "起立／蹲下切換 (0x21010202)"}
        if action in SIMPLE_CMDS:
            _burst(SIMPLE_CMDS[action], 0, 0)
            return {"ok": True, "msg": f"{action} 已送出 ({SIMPLE_CMDS[action]:#010x})"}
        return {"ok": False, "msg": "unknown action"}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


# ---------------------------------------------------------------- HTTP
PAGE = None  # set after HTML defined


def _read_infer():
    try:
        with open("/dev/shm/hydra/stats.json") as fh:
            st = json.load(fh)
        st["online"] = (time.time() - st.get("ts", 0)) < 4.0
        return st
    except Exception:
        return {"online": False}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            b = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path.startswith("/infer/") and ".jpg" in self.path:
            name = self.path.split("/infer/", 1)[1].split("?", 1)[0]
            if name not in ("frame.jpg", "bev.jpg"):
                name = "frame.jpg"
            try:
                with open("/dev/shm/hydra/" + name, "rb") as fh:
                    body = fh.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except Exception:
                self.send_response(503)
                self.end_headers()
        elif self.path.startswith("/cam"):
            rest = self.path[4:] or "/"  # keep querystring for LL-HLS
            if not rest.startswith("/"):
                rest = "/" + rest
            url = "http://127.0.0.1:8888/test" + rest
            try:
                up = urllib.request.urlopen(url, timeout=15)
                body = up.read()
                ct = up.headers.get("Content-Type", "application/octet-stream")
                self.send_response(200)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)
            except urllib.error.HTTPError as e:
                self.send_response(e.code)
                self.end_headers()
            except Exception:
                self.send_response(502)
                self.end_headers()
        elif self.path.startswith("/api/telemetry"):
            self._json(
                {
                    "t": time.time(),
                    "robot": robot_snapshot(),
                    "system": system_snapshot(),
                    "camera": {
                        "webrtc": f"http://{ROBOT_IP}:8889/test",
                        "hls": f"http://{ROBOT_IP}:8888/test/index.m3u8",
                        "path": "test",
                    },
                    "infer": _read_infer(),
                    "teleop": {"armed": _move["armed"]},
                    "battery": {"available": True, "source": "0x0901 RobotStateUpload"},
                }
            )
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path.startswith("/api/control") or self.path.startswith("/api/move"):
            ln = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(ln) if ln else b"{}"
            try:
                j = json.loads(body)
            except Exception:
                j = {}
            if self.path.startswith("/api/move"):
                self._json(
                    do_move(
                        float(j.get("vx", 0)), float(j.get("vy", 0)), float(j.get("vyaw", 0))
                    )
                )
            else:
                self._json(do_control(j.get("action", "")))
        else:
            self.send_response(404)
            self.end_headers()


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self):
        # skip HTTPServer.server_bind's socket.getfqdn() reverse-DNS, which hangs
        # for seconds when the robot's DNS is slow/unreachable.
        socketserver.TCPServer.server_bind(self)
        _host, port = self.server_address[:2]
        self.server_name = "hydra"
        self.server_port = port


def main():
    global PAGE
    PAGE = build_page()
    threading.Thread(target=telem_listener, daemon=True).start()
    threading.Thread(target=sys_sampler, daemon=True).start()
    threading.Thread(target=move_streamer, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("HydraNet dashboard on :%d" % PORT)
    srv.serve_forever()


# HTML is large; kept in a separate function for readability
def build_page():
    return HTML


HTML = r"""<!doctype html><html lang="zh-Hant"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HydraNet · Lite3 Playground</title>
<style>
:root{--bg:#0a0e14;--card:#131a24;--card2:#0f151d;--line:#232d3b;--fg:#e6edf3;--dim:#8b98a8;
 --accent:#39d0d8;--good:#3fb950;--warn:#d29922;--bad:#f85149;--mag:#bc6bff;}
*{box-sizing:border-box}
body{margin:0;font-family:"SF Pro Text",system-ui,"Noto Sans TC",sans-serif;background:var(--bg);color:var(--fg);font-size:14px}
header{display:flex;align-items:center;gap:14px;padding:12px 18px;border-bottom:1px solid var(--line);position:sticky;top:0;background:linear-gradient(180deg,#0d131b,#0a0e14);z-index:10}
header h1{font-size:16px;margin:0;font-weight:600;letter-spacing:.3px}
header .sub{color:var(--dim);font-size:12px}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:5px}
.on{background:var(--good);box-shadow:0 0 8px var(--good)}.off{background:var(--bad);box-shadow:0 0 8px var(--bad)}
.pill{margin-left:auto;font-size:12px;color:var(--dim);display:flex;gap:16px;align-items:center}
main{display:grid;grid-template-columns:repeat(12,1fr);gap:14px;padding:16px;max-width:1500px;margin:0 auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.card h2{margin:0 0 10px;font-size:12px;text-transform:uppercase;letter-spacing:1px;color:var(--dim);font-weight:600;display:flex;align-items:center;gap:8px}
.span3{grid-column:span 3}.span4{grid-column:span 4}.span5{grid-column:span 5}.span6{grid-column:span 6}.span7{grid-column:span 7}.span8{grid-column:span 8}.span12{grid-column:span 12}
@media(max-width:1100px){.span3,.span4,.span5{grid-column:span 6}.span7,.span8{grid-column:span 12}}
@media(max-width:680px){main{grid-template-columns:1fr}[class*=span]{grid-column:span 1!important}}
.big{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums}
.unit{font-size:12px;color:var(--dim);font-weight:400}
.row{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1a2230;font-variant-numeric:tabular-nums}
.row:last-child{border-bottom:0}.row .k{color:var(--dim)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:2px 18px}
.bar{height:6px;background:#1a2230;border-radius:4px;overflow:hidden;margin-top:4px}
.bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--mag));border-radius:4px;transition:width .4s}
.cores{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.core{background:var(--card2);border:1px solid var(--line);border-radius:8px;padding:6px 8px;font-size:11px}
.core b{font-size:15px;font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:12.5px}
th,td{text-align:right;padding:4px 6px;border-bottom:1px solid #1a2230}
th:first-child,td:first-child{text-align:left;color:var(--dim)}
th{color:var(--dim);font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:.5px}
.cam{position:relative;width:100%;aspect-ratio:16/9;background:#000;border-radius:10px;overflow:hidden;border:1px solid var(--line)}
.cam iframe{width:100%;height:100%;border:0}
.pose{display:flex;gap:18px;align-items:center;flex-wrap:wrap}
.scene{width:150px;height:150px;perspective:520px;flex:0 0 auto;margin:auto}
.dog{width:118px;height:64px;transform-style:preserve-3d;transition:transform .12s linear;margin:43px auto;position:relative}
.face{position:absolute;border:1.5px solid var(--accent);background:rgba(57,208,216,.10)}
.controls{display:flex;flex-direction:column;gap:12px}
.estop{width:100%;padding:18px;border:0;border-radius:12px;background:var(--bad);color:#fff;font-size:18px;font-weight:800;letter-spacing:2px;cursor:pointer;box-shadow:0 0 0 rgba(248,81,73,.6);animation:none}
.estop:active{transform:scale(.98)}
.posture{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.btn{padding:14px;border-radius:10px;border:1px solid var(--line);background:var(--card2);color:var(--fg);font-size:15px;font-weight:600;cursor:pointer;position:relative}
.btn:hover{border-color:var(--accent)}.btn:active{transform:scale(.98)}
.btn.pending::after{content:"待擷取";position:absolute;top:6px;right:8px;font-size:9px;color:var(--warn);border:1px solid var(--warn);border-radius:4px;padding:0 4px;font-weight:600}
.dpad{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;max-width:230px;margin:auto}
.dpad .btn{padding:12px;font-size:16px}
.na{color:var(--dim);font-size:12px;line-height:1.5;background:var(--card2);border:1px dashed var(--line);border-radius:8px;padding:10px}
.toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#1c2633;border:1px solid var(--line);padding:10px 16px;border-radius:8px;font-size:13px;opacity:0;transition:opacity .3s;pointer-events:none;max-width:90vw;z-index:50}
.toast.show{opacity:1}
.mini{font-size:11px;color:var(--dim)}
.tag{font-size:10px;padding:1px 6px;border-radius:5px;border:1px solid var(--line);color:var(--dim)}
.warnc{color:var(--warn)}.badc{color:var(--bad)}.goodc{color:var(--good)}
</style></head><body>
<header>
  <div><h1>🐾 HydraNet · Lite3 Playground</h1><div class="sub" id="hostline">connecting…</div></div>
  <div class="pill">
    <span><span class="dot off" id="rdot"></span><span id="rstate">robot</span></span>
    <span id="ratehz">— Hz</span>
    <span id="clock"></span>
  </div>
</header>
<main>
  <!-- POSE -->
  <section class="card span5">
    <h2>姿態 · IMU</h2>
    <div class="pose">
      <div class="scene"><div class="dog" id="dog">
        <div class="face" style="width:118px;height:64px;transform:translateZ(16px)"></div>
        <div class="face" style="width:118px;height:64px;transform:translateZ(-16px)"></div>
        <div class="face" style="width:32px;height:64px;left:0;transform:rotateY(90deg) translateZ(0px)"></div>
        <div class="face" style="width:32px;height:64px;right:0;transform:rotateY(90deg) translateZ(102px)"></div>
        <div class="face" style="width:118px;height:32px;top:0;transform:rotateX(90deg) translateZ(0px)"></div>
        <div class="face" style="width:118px;height:32px;bottom:0;transform:rotateX(90deg) translateZ(-64px);background:rgba(188,107,255,.12);border-color:var(--mag)"></div>
      </div></div>
      <div style="flex:1;min-width:150px">
        <div class="grid2">
          <div class="row"><span class="k">Roll</span><span id="roll">—</span></div>
          <div class="row"><span class="k">ω roll</span><span id="wroll">—</span></div>
          <div class="row"><span class="k">Pitch</span><span id="pitch">—</span></div>
          <div class="row"><span class="k">ω pitch</span><span id="wpitch">—</span></div>
          <div class="row"><span class="k">Yaw</span><span id="yaw">—</span></div>
          <div class="row"><span class="k">ω yaw</span><span id="wyaw">—</span></div>
        </div>
        <div class="grid2" style="margin-top:8px">
          <div class="row"><span class="k">acc X</span><span id="ax">—</span></div>
          <div class="row"><span class="k">acc Y</span><span id="ay">—</span></div>
          <div class="row"><span class="k">acc Z</span><span id="az">—</span></div>
          <div class="row"><span class="k">tick</span><span id="tick">—</span></div>
        </div>
        <div class="mini" style="margin-top:6px">角度單位 °，角速度 °/s，加速度 m/s²　·　來源 MotionSDK 0x0906 @43897（被動唯讀）</div>
      </div>
    </div>
  </section>

  <!-- CAMERA -->
  <section class="card span7">
    <h2>視覺 · 前置相機 <span class="tag" id="camtag">MediaMTX HLS</span></h2>
    <div class="cam"><iframe id="camframe" allow="autoplay" referrerpolicy="no-referrer"></iframe></div>
    <div class="mini" style="margin-top:6px">/dev/video0 · USB MJPEG 1280×720@30 · H264 · 串流 <a id="hlslink" style="color:var(--accent)" target="_blank">HLS</a> · <a id="wrtclink" style="color:var(--accent)" target="_blank">WebRTC 低延遲(同網段)</a></div>
  </section>

  <!-- INFERENCE -->
  <section class="card span12">
    <h2>HydraNet 推論（NPU） <span class="tag" id="inftag">—</span></h2>
    <div style="display:flex;gap:16px;flex-wrap:wrap">
      <div style="flex:2;min-width:320px">
        <div class="cam"><img id="infimg" style="width:100%;height:100%;object-fit:contain;background:#000" alt="inference"></div>
      </div>
      <div style="flex:1;min-width:170px">
        <div class="mini" style="margin-bottom:4px">俯視 BEV（單目近似）</div>
        <div style="width:100%;aspect-ratio:26/30;background:#000;border-radius:8px;overflow:hidden;border:1px solid var(--line)">
          <img id="bevimg" style="width:100%;height:100%;object-fit:contain" alt="bev"></div>
      </div>
      <div style="flex:1;min-width:170px">
        <div class="row"><span class="k">FPS</span><span id="inf_fps">—</span></div>
        <div class="row"><span class="k">推論延遲</span><span id="inf_ms">—</span></div>
        <div class="row"><span class="k">偵測數</span><span id="inf_ndet">—</span></div>
        <div class="mini" style="margin-top:8px">可通行 traversability</div>
        <div class="row"><span class="k goodc">go 可走</span><span id="inf_go">—</span></div>
        <div class="row"><span class="k warnc">caution</span><span id="inf_caution">—</span></div>
        <div class="row"><span class="k badc">blocked 阻擋</span><span id="inf_blocked">—</span></div>
        <div class="mini" id="inf_counts" style="margin-top:8px"></div>
      </div>
    </div>
    <div class="mini" style="margin-top:6px">綠=可通行 · 紅=阻擋 · 黃框=COCO 偵測　·　hydranet_joint_coco10 fp16 @ RK3588 NPU（三頭多任務）</div>
  </section>

  <!-- CONTROL -->
  <section class="card span5">
    <h2>操控 <span class="tag warnc">會移動真機</span> <span class="tag" id="armtag">未啟用</span></h2>
    <div class="controls">
      <button class="estop" id="estop">■ 緊急停止 E-STOP</button>
      <button class="btn" id="armbtn" style="width:100%;font-size:16px;padding:14px;border-color:var(--warn)">🔓 啟用操控（移動＋手動模式）</button>
      <div class="mini">移動（按住＝走、放開＝停；鍵盤 W/S 前後 · A/D 平移 · Q/E 轉向）</div>
      <div class="dpad">
        <span></span><button class="btn mv" data-vx="1">▲ 前</button><span></span>
        <button class="btn mv" data-vy="-1">◀ 左移</button>
        <button class="btn mv" data-vyaw="-1">↺ 左轉</button>
        <button class="btn mv" data-vy="1">右移 ▶</button>
        <button class="btn mv" data-vyaw="1">↻ 右轉</button>
        <button class="btn mv" data-vx="-1">▼ 後</button>
        <span></span>
      </div>
      <div class="mini">姿態 · 步態 · 動作</div>
      <div class="posture" style="grid-template-columns:1fr 1fr">
        <button class="btn" data-a="toggle">🦿 起立⇄趴下</button>
        <button class="btn" data-a="zero">↺ 回零</button>
      </div>
      <div class="posture" style="grid-template-columns:repeat(3,1fr)">
        <button class="btn" data-a="gait_low">低速</button>
        <button class="btn" data-a="gait_mid">中速</button>
        <button class="btn" data-a="gait_high">高速</button>
        <button class="btn" data-a="gait_grip">抓地越障</button>
        <button class="btn" data-a="gait_highstep">高踏步</button>
        <button class="btn" data-a="act_greet">打招呼</button>
      </div>
      <div class="na" id="ctlnote"><b>會移動真機——操作前確保周圍淨空、硬體急停待命。</b><br>
      官方通訊接口 V1.0.7：急停 <code>0x21020C0E</code>；移動軸指令 @50Hz 串流，機器人 250ms
      未收到即自停（deadman），放開即停；<code>0x2101013x</code> 前後/平移/轉向。</div>
    </div>
  </section>

  <!-- BATTERY + SENSORS (0x0901) -->
  <section class="card span3">
    <h2>電量 · 感測 (0x0901)</h2>
    <div class="big" id="battbig">—</div>
    <div class="bar"><i id="battbar" style="width:0"></i></div>
    <div class="row" style="margin-top:8px"><span class="k">超音波 L / R</span><span id="ultra">—</span></div>
    <div class="row"><span class="k">位姿 x/y/z (m)</span><span id="posw">—</span></div>
    <div class="row"><span class="k">body 速度 (m/s)</span><span id="velb">—</span></div>
    <div class="mini" style="margin-top:6px" id="battnote">RobotStateUpload @50Hz · 官方遙測</div>
  </section>

  <section class="card span4">
    <h2>主機 · SoC</h2>
    <div id="board" class="big" style="font-size:16px;line-height:1.4">—</div>
    <div class="row" style="margin-top:8px"><span class="k">Hostname</span><span id="hn">—</span></div>
    <div class="row"><span class="k">Kernel</span><span id="kern">—</span></div>
    <div class="row"><span class="k">Uptime</span><span id="uptime">—</span></div>
    <div class="row"><span class="k">Load (1/5/15m)</span><span id="load">—</span></div>
  </section>

  <section class="card span5">
    <h2>溫度 · 熱區</h2>
    <div id="temps" class="grid2"></div>
  </section>

  <!-- CPU CORES -->
  <section class="card span5">
    <h2>CPU · 每核心</h2>
    <div class="cores" id="cores"></div>
  </section>

  <!-- MEM/DISK/NET -->
  <section class="card span3">
    <h2>記憶體 · 儲存</h2>
    <div class="mini">RAM</div><div class="big" id="memv">—</div><div class="bar"><i id="membar" style="width:0"></i></div>
    <div class="mini" style="margin-top:12px">Disk /</div><div class="big" id="diskv">—</div><div class="bar"><i id="diskbar" style="width:0"></i></div>
  </section>

  <section class="card span4">
    <h2>網路</h2>
    <div id="nets"></div>
    <div class="row" style="margin-top:6px"><span class="k">Wi-Fi 訊號</span><span id="wifi">—</span></div>
  </section>

  <!-- JOINTS -->
  <section class="card span12">
    <h2>12 關節 · 馬達 <span class="mini" id="jhz"></span></h2>
    <div style="overflow-x:auto"><table id="jtable">
      <thead><tr><th>關節</th><th>位置 rad</th><th>速度 rad/s</th><th>力矩 Nm</th><th>溫度 °C</th></tr></thead>
      <tbody></tbody></table></div>
    <div class="mini" style="margin-top:6px">溫度欄由 jy_exe 遙測回報（目前回 0，代表該欄未提供）。</div>
  </section>
</main>
<div class="toast" id="toast"></div>
<script>
const $=id=>document.getElementById(id);
function fmt(v,d=2){return (v==null||isNaN(v))?"—":Number(v).toFixed(d)}
function toast(m){const t=$("toast");t.textContent=m;t.classList.add("show");clearTimeout(t._h);t._h=setTimeout(()=>t.classList.remove("show"),3200)}
function human(b){if(!b)return"—";const u=["B","KB","MB","GB","TB"];let i=0;while(b>=1024&&i<u.length-1){b/=1024;i++}return b.toFixed(1)+" "+u[i]}
function dur(s){if(s==null)return"—";const d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60);return (d?d+"d ":"")+h+"h "+m+"m"}

let camSet=false;
async function tick(){
  try{
    const r=await fetch("/api/telemetry",{cache:"no-store"});const j=await r.json();
    // robot
    const rb=j.robot;
    $("rdot").className="dot "+(rb.online?"on":"off");
    $("rstate").textContent=rb.online?"robot online":"robot offline";
    $("ratehz").textContent=(rb.hz||0)+" Hz";
    if(rb.imu){const m=rb.imu;
      $("roll").textContent=fmt(m.roll)+"°";$("pitch").textContent=fmt(m.pitch)+"°";$("yaw").textContent=fmt(m.yaw)+"°";
      $("wroll").textContent=fmt(m.wroll,1);$("wpitch").textContent=fmt(m.wpitch,1);$("wyaw").textContent=fmt(m.wyaw,1);
      $("ax").textContent=fmt(m.ax);$("ay").textContent=fmt(m.ay);$("az").textContent=fmt(m.az);
      $("tick").textContent=rb.tick;
      $("dog").style.transform=`rotateX(${-m.pitch}deg) rotateZ(${m.roll}deg) rotateY(${m.yaw}deg)`;
    }
    if(rb.joints){const tb=$("jtable").querySelector("tbody");tb.innerHTML="";
      for(const jt of rb.joints){const tr=document.createElement("tr");
        const warm=jt.temp>55?' class="warnc"':'';
        tr.innerHTML=`<td>${jt.name}</td><td>${fmt(jt.pos,3)}</td><td>${fmt(jt.vel,3)}</td><td>${fmt(jt.tau,3)}</td><td${warm}>${fmt(jt.temp,1)}</td>`;
        tb.appendChild(tr);}
      $("jhz").textContent="· "+(rb.hz||0)+" Hz · age "+(rb.age_ms==null?"—":rb.age_ms+"ms");
    }
    // camera
    if(!camSet){const h=location.hostname;$("camframe").src="/cam/";$("hlslink").href="/cam/";$("wrtclink").href="http://"+h+":8889/test/";camSet=true;}
    if(j.teleop&&j.teleop.armed!==armed){armed=j.teleop.armed;updateArm();}
    if(j.infer){const q=j.infer;const tg=$("inftag");tg.textContent=q.online?"● live":"○ offline";tg.style.color=q.online?"var(--good)":"var(--dim)";
      $("inf_fps").textContent=(q.fps||0)+" FPS";$("inf_ms").textContent=(q.infer_ms||0)+" ms";$("inf_ndet").textContent=q.n_det||0;
      if(q.trav){$("inf_go").textContent=(q.trav.go*100).toFixed(0)+"%";$("inf_caution").textContent=(q.trav.caution*100).toFixed(0)+"%";$("inf_blocked").textContent=(q.trav.blocked*100).toFixed(0)+"%";}
      $("inf_counts").textContent=Object.entries(q.det_counts||{}).map(function(e){return e[0]+"×"+e[1]}).join("   ");}
    // battery + sensors (0x0901)
    const rb2=j.robot||{};
    if(rb2.battery!=null){const b=rb2.battery;$("battbig").innerHTML=b.toFixed(0)+'<span class="unit">%</span>';$("battbig").style.color=b>50?"var(--good)":b>20?"var(--warn)":"var(--bad)";if($("battbar"))$("battbar").style.width=Math.max(0,Math.min(100,b))+"%";}
    if(rb2.ultrasound){$("ultra").textContent=rb2.ultrasound.map(function(x){return x.toFixed(2)+"m"}).join(" / ");}
    if(rb2.pos_world){$("posw").textContent=rb2.pos_world.map(function(x){return x.toFixed(2)}).join(", ");}
    if(rb2.vel_body){$("velb").textContent=rb2.vel_body.map(function(x){return x.toFixed(2)}).join(", ");}
    // system
    const s=j.system;
    if(s.static){$("board").textContent=s.static.model;$("hn").textContent=s.static.hostname;$("kern").textContent=s.static.kernel;$("hostline").textContent=s.static.model+" · "+s.static.hostname;}
    $("uptime").textContent=dur(s.uptime_s);
    if(s.load)$("load").textContent=s.load.join(" / ");
    if(s.cores){const c=$("cores");c.innerHTML="";for(const co of s.cores){const d=document.createElement("div");d.className="core";
      const col=co.use>85?"var(--bad)":co.use>60?"var(--warn)":"var(--good)";
      d.innerHTML=`<div class="mini">core ${co.core}</div><b style="color:${col}">${fmt(co.use,0)}%</b><div class="mini">${co.mhz?co.mhz+" MHz":""}</div>`;c.appendChild(d);}}
    if(s.temps){const t=$("temps");t.innerHTML="";for(const tp of s.temps){const col=tp.c>75?"badc":tp.c>60?"warnc":"";
      t.innerHTML+=`<div class="row"><span class="k">${tp.name}</span><span class="${col}">${fmt(tp.c,1)}°C</span></div>`;}}
    if(s.mem){$("memv").innerHTML=`${fmt(s.mem.used_pct,0)}% <span class="unit">${human(s.mem.total-s.mem.avail)} / ${human(s.mem.total)}</span>`;$("membar").style.width=s.mem.used_pct+"%";}
    if(s.disk){$("diskv").innerHTML=`${fmt(s.disk.used_pct,0)}% <span class="unit">${human(s.disk.total-s.disk.free)} / ${human(s.disk.total)}</span>`;$("diskbar").style.width=s.disk.used_pct+"%";}
    if(s.net){const n=$("nets");n.innerHTML="";for(const nt of s.net){n.innerHTML+=`<div class="row"><span class="k">${nt.iface} <span class="tag">${nt.state}</span></span><span>${nt.addr}</span></div>`;}}
    if(s.wifi)$("wifi").textContent=`${s.wifi.iface} · ${s.wifi.level_dbm} dBm`;
  }catch(e){$("rstate").textContent="dashboard error";}
}
async function ctl(action){
  try{const r=await fetch("/api/control",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({action})});
    const j=await r.json();toast(j.msg||(j.ok?"ok":"failed"));return j;}
  catch(e){toast("送出失敗："+e);return{}}
}
let armed=false;
function updateArm(){$("armtag").textContent=armed?"● 已啟用":"未啟用";$("armtag").style.color=armed?"var(--good)":"var(--dim)";var b=$("armbtn");b.textContent=armed?"🔒 停用操控":"🔓 啟用操控（移動＋手動模式）";b.style.borderColor=armed?"var(--good)":"var(--warn)";}
$("armbtn").onclick=async function(){const j=await ctl(armed?"disarm":"arm");if(j&&"armed"in j){armed=j.armed;}else{armed=!armed;}updateArm();};
$("estop").onclick=function(){armed=false;updateArm();ctl("estop");};
document.querySelectorAll(".btn[data-a]").forEach(function(b){b.onclick=function(){ctl(b.dataset.a);};});
// --- movement: held buttons + WASDQE keys -> velocity vector, streamed while armed ---
const mv={vx:0,vy:0,vyaw:0};const held=new Set();const keyHeld={};
function clamp(x){return Math.max(-1,Math.min(1,x));}
function recompute(){let vx=0,vy=0,vyaw=0;
  held.forEach(function(b){vx+=+(b.dataset.vx||0);vy+=+(b.dataset.vy||0);vyaw+=+(b.dataset.vyaw||0);});
  Object.values(keyHeld).forEach(function(k){vx+=k.vx||0;vy+=k.vy||0;vyaw+=k.vyaw||0;});
  mv.vx=clamp(vx);mv.vy=clamp(vy);mv.vyaw=clamp(vyaw);}
setInterval(function(){if(!armed)return;fetch("/api/move",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(mv)}).catch(function(){});},100);
document.querySelectorAll(".mv").forEach(function(b){
  const press=function(e){if(e)e.preventDefault();if(!armed){toast("請先啟用操控");return;}held.add(b);b.style.borderColor="var(--good)";recompute();};
  const rel=function(){held.delete(b);b.style.borderColor="";recompute();};
  b.addEventListener("mousedown",press);b.addEventListener("touchstart",press,{passive:false});
  ["mouseup","mouseleave","touchend","touchcancel"].forEach(function(ev){b.addEventListener(ev,rel);});});
const KEYMAP={KeyW:{vx:1},KeyS:{vx:-1},KeyA:{vy:-1},KeyD:{vy:1},KeyQ:{vyaw:-1},KeyE:{vyaw:1}};
document.addEventListener("keydown",function(e){if(KEYMAP[e.code]&&armed&&!keyHeld[e.code]){keyHeld[e.code]=KEYMAP[e.code];recompute();}});
document.addEventListener("keyup",function(e){if(KEYMAP[e.code]){delete keyHeld[e.code];recompute();}});
setInterval(()=>{$("clock").textContent=new Date().toLocaleTimeString()},1000);
setInterval(function(){var t=Date.now();var im=$("infimg");if(im)im.src="/infer/frame.jpg?t="+t;var bv=$("bevimg");if(bv)bv.src="/infer/bev.jpg?t="+t;},300);
tick();setInterval(tick,500);
</script>
</body></html>
"""

if __name__ == "__main__":
    main()
