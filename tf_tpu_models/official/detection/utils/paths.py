import os


def root_dir(path: str = ''):
    # return the path if it's already absolute
    if path and (is_bucket_path(path) or os.path.isabs(path)):
        return path

    res_path = os.path.abspath(os.path.join(os.path.dirname(__file__), *['..'] * 4))
    if path:
        res_path = os.path.join(res_path, path)

    return res_path


def is_bucket_path(path: str):
    return path.startswith('gs://') or path.startswith('s3://')
