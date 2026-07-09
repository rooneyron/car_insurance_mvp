"""
计时工具
用于性能监控，记录各阶段耗时。
"""

import time


class Timer:
    def __init__(self):
        self.logs = []

    def start(self, label: str):
        self.logs.append({"label": label, "start": time.time(), "end": None})

    def stop(self, label: str = None):
        if label:
            for log in self.logs:
                if log["label"] == label and log["end"] is None:
                    log["end"] = time.time()
                    break
        else:
            for log in reversed(self.logs):
                if log["end"] is None:
                    log["end"] = time.time()
                    break

    def get_report(self) -> str:
        report = []
        for log in self.logs:
            if log["end"] is not None:
                elapsed = (log["end"] - log["start"]) * 1000
                report.append(f"  ├── {log['label']}: {elapsed:.0f}ms")
        return "\n".join(report)

    def get_total_ms(self) -> float:
        if self.logs and self.logs[-1]["end"] is not None:
            return (self.logs[-1]["end"] - self.logs[0]["start"]) * 1000
        return 0
