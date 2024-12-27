import time

from functools import wraps


def monitor_prediction_time():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            result = func(*args, **kwargs)

            end_time = time.time()

            print(f"Prediction time: {end_time - start_time:.4f} seconds")

            return result

        return wrapper

    return decorator
