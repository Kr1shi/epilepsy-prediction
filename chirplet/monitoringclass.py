import pynvml
import psutil
import time
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from multiprocessing import Process, Event, Manager

class MonitoringClass:
    def __init__(self, interval=0.1):
        self.interval = interval

        self.stop_cpu_flag = Event()
        self.stop_gpu_flag = Event()

        self.manager = Manager()
        self.CPU_usage = self.manager.list()
        self.Memory_usage = self.manager.list()
        self.counter = self.manager.Value('i', 0)

    def _CPU_monitoring(self, stop_flag, interval, CPU_usage, Memory_usage, counter):
        while not stop_flag.is_set():
            cpu_usage = psutil.cpu_percent(0)
            memory_usage = psutil.virtual_memory().used / 1024**3

            CPU_usage.append(cpu_usage)
            Memory_usage.append(memory_usage)

            counter.value += 1
            print(f"[{datetime.now().isoformat()}] CPU sample #{counter.value}")

            with open('cpu_monitoring.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([cpu_usage, memory_usage])
                file.flush()

            time.sleep(interval)

    def _GPU_monitoring(self, stop_flag, interval):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        while not stop_flag.is_set():
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            power = pynvml.nvmlDeviceGetPowerUsage(handle)
            usage = pynvml.nvmlDeviceGetUtilizationRates(handle)

            with open('gpu_monitoring.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([usage.gpu, memory.used / 1024**3, power / 1000])
                file.flush()

            time.sleep(interval)

    def start_CPU_monitoring(self):
        self.cpu_proc = Process(
            target=self._CPU_monitoring,
            args=(self.stop_cpu_flag, self.interval, self.CPU_usage, self.Memory_usage, self.counter)
        )
        self.cpu_proc.start()

    def stop_CPU_monitoring(self):
        self.stop_cpu_flag.set()
        self.cpu_proc.join()
        print("Stopped CPU monitoring.")

    def start_GPU_monitoring(self):
        self.gpu_proc = Process(
            target=self._GPU_monitoring,
            args=(self.stop_gpu_flag, self.interval)
        )
        self.gpu_proc.start()

    def stop_GPU_monitoring(self):
        self.stop_gpu_flag.set()
        self.gpu_proc.join()
        print("Stopped GPU monitoring.")

# if __name__ == "__main__":
#     monitor = MonitoringClass()
#     monitor.start_CPU_monitoring()
#     monitor.start_GPU_monitoring()

#     time.sleep(15)  # Simulate work

#     monitor.stop_CPU_monitoring()
#     monitor.stop_GPU_monitoring()

#     print(f"Total samples: {monitor.counter.value}")
#     plt.plot(monitor.CPU_usage)
#     plt.show()