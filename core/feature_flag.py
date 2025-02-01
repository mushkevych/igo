import os
import threading
import time

UPDATE_INTERVAL_SEC = 300  # 5 minutes
FEATURE_FLAG_PREFIX = 'feature_flag_'


class FeatureFlagMgr:
    def __init__(self):
        self.flags: dict[str, bool] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        """Starts a background thread to periodically update the feature flags."""
        def updater():
            while True:
                self._update_flags_from_env()
                time.sleep(UPDATE_INTERVAL_SEC)

        thread = threading.Thread(target=updater, daemon=True)
        thread.start()

    def _update_flags_from_env(self) -> None:
        """Reads environment variables with 'feature_flag_' prefix and updates the flags dict."""
        with self._lock:
            for key, value in os.environ.items():
                lower_key = key.lower()
                if lower_key.startswith(FEATURE_FLAG_PREFIX):
                    feature_name = lower_key.split(FEATURE_FLAG_PREFIX, 1)[1]
                    self.flags[feature_name] = self._as_boolean(value)

    def _as_boolean(self, value: str) -> bool:
        """Converts string values to boolean based on specific criteria."""
        value = str(value).lower()
        if value in ('true', '1'):
            return True
        elif value in ('false', '0'):
            return False
        else:
            return False

    def is_enabled(self, feature_name: str) -> bool:
        """Checks if the feature flag is enabled. Returns False if the flag is not found."""
        with self._lock:
            return self.flags.get(feature_name, False)
