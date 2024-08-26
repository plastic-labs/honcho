import time
import csv
from functools import wraps
from datetime import datetime
import os

csv_file_path = 'timing_logs.csv'

def timing_decorator(csv_file):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not kwargs.get('enable_timing', False):
                return await func(*args, **kwargs)
            
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            session_id = kwargs.get('session_id')
            
            data = {
                'session_id': str(session_id),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'function': func.__name__,
                'execution_time': execution_time
            }
            
            file_exists = os.path.isfile(csv_file)
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)
            
            return result
        return wrapper
    return decorator