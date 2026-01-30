"""
Health Monitoring and Metrics for Football Pipeline
Tracks processing status, performance, and system health.
"""

import time
import json
import os
from collections import defaultdict, deque
from datetime import datetime
import threading


class HealthMonitor:
    """
    Centralized health monitoring and metrics collection.
    Thread-safe singleton for pipeline-wide observability.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.start_time = time.time()
        self.metrics = {
            'videos_processed': 0,
            'videos_failed': 0,
            'videos_running': 0,
            'total_frames_processed': 0,
            'total_processing_time_s': 0,
            'db_queries': 0,
            'db_errors': 0,
            'api_calls': 0,
            'api_errors': 0,
        }

        # Performance tracking (last 100 videos)
        self.processing_times = deque(maxlen=100)
        self.frame_rates = deque(maxlen=100)

        # Error tracking
        self.errors = deque(maxlen=50)

        # Active jobs
        self.active_jobs = {}

        self._initialized = True

    def record_video_start(self, video_id, user_id=None):
        """Record start of video processing."""
        with self._lock:
            self.active_jobs[video_id] = {
                'start_time': time.time(),
                'user_id': user_id,
                'status': 'running',
                'frames': 0
            }
            self.metrics['videos_running'] += 1

    def record_video_complete(self, video_id, frames_processed=0):
        """Record successful video completion."""
        with self._lock:
            if video_id in self.active_jobs:
                job = self.active_jobs[video_id]
                duration = time.time() - job['start_time']

                self.metrics['videos_processed'] += 1
                self.metrics['videos_running'] -= 1
                self.metrics['total_frames_processed'] += frames_processed
                self.metrics['total_processing_time_s'] += duration

                self.processing_times.append(duration)
                if frames_processed > 0 and duration > 0:
                    self.frame_rates.append(frames_processed / duration)

                del self.active_jobs[video_id]

    def record_video_failure(self, video_id, error_msg):
        """Record video processing failure."""
        with self._lock:
            if video_id in self.active_jobs:
                self.metrics['videos_failed'] += 1
                self.metrics['videos_running'] -= 1

                self.errors.append({
                    'timestamp': datetime.now().isoformat(),
                    'video_id': video_id,
                    'error': str(error_msg)
                })

                del self.active_jobs[video_id]

    def record_db_query(self, success=True):
        """Record database query."""
        with self._lock:
            self.metrics['db_queries'] += 1
            if not success:
                self.metrics['db_errors'] += 1

    def record_api_call(self, success=True):
        """Record API call."""
        with self._lock:
            self.metrics['api_calls'] += 1
            if not success:
                self.metrics['api_errors'] += 1

    def get_health_status(self):
        """
        Get current health status.
        Returns: dict with health metrics and status.
        """
        with self._lock:
            uptime_s = time.time() - self.start_time

            # Calculate averages
            avg_processing_time = (
                sum(self.processing_times) / len(self.processing_times)
                if self.processing_times else 0
            )

            avg_frame_rate = (
                sum(self.frame_rates) / len(self.frame_rates)
                if self.frame_rates else 0
            )

            # Determine overall health
            health = 'healthy'
            if self.metrics['videos_running'] > 10:
                health = 'degraded'  # Too many concurrent jobs
            elif self.metrics['db_errors'] > 5:
                health = 'unhealthy'  # DB issues
            elif self.metrics['videos_failed'] > self.metrics['videos_processed']:
                health = 'unhealthy'  # More failures than successes

            return {
                'status': health,
                'uptime_seconds': int(uptime_s),
                'uptime_hours': round(uptime_s / 3600, 2),
                'metrics': dict(self.metrics),
                'performance': {
                    'avg_processing_time_s': round(avg_processing_time, 2),
                    'avg_frame_rate_fps': round(avg_frame_rate, 2),
                    'active_jobs': len(self.active_jobs),
                    'recent_errors': len(self.errors)
                },
                'active_jobs': {
                    vid: {
                        'duration_s': int(time.time() - job['start_time']),
                        'status': job['status']
                    }
                    for vid, job in self.active_jobs.items()
                },
                'recent_errors': list(self.errors)[-5:]  # Last 5 errors
            }

    def save_snapshot(self, filepath='output/health_snapshot.json'):
        """Save health snapshot to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.get_health_status(), f, indent=2)
        print(f"[health] Snapshot saved to {filepath}")

    def print_summary(self):
        """Print health summary to console."""
        status = self.get_health_status()

        print("\n" + "="*60)
        print("PIPELINE HEALTH SUMMARY")
        print("="*60)
        print(f"Status: {status['status'].upper()}")
        print(f"Uptime: {status['uptime_hours']} hours")
        print(f"\nProcessing:")
        print(f"  ✅ Completed: {status['metrics']['videos_processed']}")
        print(f"  ❌ Failed: {status['metrics']['videos_failed']}")
        print(f"  🔄 Running: {status['metrics']['videos_running']}")
        print(f"  📊 Total Frames: {status['metrics']['total_frames_processed']:,}")
        print(f"\nPerformance:")
        print(f"  ⏱️  Avg Processing Time: {status['performance']['avg_processing_time_s']}s")
        print(f"  🎬 Avg Frame Rate: {status['performance']['avg_frame_rate_fps']} fps")
        print(f"\nDatabase:")
        print(f"  📤 Queries: {status['metrics']['db_queries']}")
        print(f"  ⚠️  Errors: {status['metrics']['db_errors']}")
        print(f"\nAPI:")
        print(f"  📡 Calls: {status['metrics']['api_calls']}")
        print(f"  ⚠️  Errors: {status['metrics']['api_errors']}")

        if status['recent_errors']:
            print(f"\nRecent Errors:")
            for err in status['recent_errors']:
                print(f"  - [{err['timestamp']}] {err['video_id']}: {err['error'][:80]}")

        print("="*60 + "\n")


# Global instance
health_monitor = HealthMonitor()


def get_health_monitor():
    """Get the global health monitor instance."""
    return health_monitor
