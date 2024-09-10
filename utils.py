import time
import boto3
from functools import wraps


def find_intersection_smaller(a1, a2, b1, b2):
    """
        finds IOU in 1D relative to size of smaller obj
    """
    min1 = max(a1, b1)
    max2 = min(a2, b2)
    if min1 > max2:
        return 0
    intersection = max2-min1
    min_width = min((a2-a1), (b2-b1))
    iou = intersection / min_width * 100
    return iou


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def download_from_s3(filename, bucket_name, s3_filename, s3_client=None):
    if not s3_client:
        s3_client = boto3.client('s3')
    s3_client.download_file(bucket_name, s3_filename, filename)
    print(f'Downloaded file {s3_filename} from S3 bucket {bucket_name}')


def upload_to_s3(filename, bucket_name, s3_filename):
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(filename, bucket_name, s3_filename)
    except Exception as e:
        print('Error uploading file to S3:', str(e))


def flatten_list(original_list):

    flattened_list = []
    for i in original_list:
        if isinstance(i, list):
            flattened_list.extend(i)
        else:
            flattened_list.append(i)

    return flattened_list
