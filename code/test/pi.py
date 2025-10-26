# testing raspberry pi by checking system information
import platform
import os
import time
import subprocess

def get_info():
    info = {
        "system": platform.system(),
        "node": platform.node(),
        "release": platform.release(),
        "python": platform.python_version()
    }
    return info

def cpu_info():
    try:
        with open("/proc/cpuinfo", "r") as cpu:
            info = "".join(cpu.readlines())
        return info
    except FileNotFoundError:
        return "CPU info not available"

def network_info():
    try:
        response = subprocess.run(
            ["ping", "-c", "1", "8.8.8.8"],
            capture_output = True,
            text=True
        )
        return response.returncode == 0
    except Exception:
        return False

def main():
    print("System info:")
    sys_info = get_info()
    for key, value in sys_info.items():
        print(f"{key}: {value}")
    
    print("\nCPU info:")
    print(cpu_info())

    print("\nNetwork:")
    if network_info():
        print("Ping successful!")
    else:
        print("Ping failed.")

if __name__ == "__main__":
    main()