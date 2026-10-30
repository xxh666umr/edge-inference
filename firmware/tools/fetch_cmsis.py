"""下载 STM32F103（ZET6）所需的 CMSIS 头文件与启动文件（通过 GitHub API）。

用法：python firmware/tools/fetch_cmsis.py
输出：firmware/cmsis/...
"""
import base64
import json
import os
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "firmware", "cmsis")

FILES = [
    ("STMicroelectronics/cmsis_device_f1", "master",
     "Include/stm32f1xx.h", "stm32f1xx.h"),
    ("STMicroelectronics/cmsis_device_f1", "master",
     "Include/stm32f103xe.h", "stm32f103xe.h"),
    ("STMicroelectronics/cmsis_device_f1", "master",
     "Include/system_stm32f1xx.h", "system_stm32f1xx.h"),
    ("STMicroelectronics/cmsis_device_f1", "master",
     "Source/Templates/system_stm32f1xx.c", "system_stm32f1xx.c"),
    ("STMicroelectronics/cmsis_device_f1", "master",
     "Source/Templates/gcc/startup_stm32f103xe.s", "startup_stm32f103xe.s"),
    ("ARM-software/CMSIS_5", "develop",
     "CMSIS/Core/Include/core_cm3.h", "core_cm3.h"),
    ("ARM-software/CMSIS_5", "develop",
     "CMSIS/Core/Include/cmsis_version.h", "cmsis_version.h"),
    ("ARM-software/CMSIS_5", "develop",
     "CMSIS/Core/Include/cmsis_compiler.h", "cmsis_compiler.h"),
    ("ARM-software/CMSIS_5", "develop",
     "CMSIS/Core/Include/cmsis_gcc.h", "cmsis_gcc.h"),
    ("ARM-software/CMSIS_5", "develop",
     "CMSIS/Core/Include/mpu_armv7.h", "mpu_armv7.h"),
]


def fetch(repo, ref, path, out_name):
    url = (f"https://api.github.com/repos/{repo}/contents/"
           f"{urllib.parse.quote(path)}?ref={ref}")
    req = urllib.request.Request(url, headers={"User-Agent": "edge-inference"})
    data = json.load(urllib.request.urlopen(req, timeout=60))
    content = base64.b64decode(data["content"])
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, out_name)
    with open(out, "wb") as f:
        f.write(content)
    print(f"{out_name}: {len(content)} bytes")


def main():
    for repo, ref, path, name in FILES:
        for attempt in range(3):
            try:
                fetch(repo, ref, path, name)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print("retry", name, e)


if __name__ == "__main__":
    main()
