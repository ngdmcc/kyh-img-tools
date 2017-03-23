import os
__author__ = 'Kyle Hounslow'

def create_empty_dirtree(srcdir, dstdir, onerror=None):
    """
    Duplicates a source directory structure (folders only!)
    :param srcdir: root of directory structure to be duplicated
    :param dstdir: root of where duplicated directory will be saved to
    :param onerror: used for os.walk() function
    :return:
    """
    srcdir = os.path.abspath(srcdir)
    srcdir_prefix = len(srcdir) + len(os.path.sep)
    try:
        os.mkdir(srcdir)
    except OSError as e:
        if onerror is not None:
            onerror(e)
    for root, dirs, files in os.walk(srcdir, onerror=onerror):
        for dirname in dirs:
            dirpath = os.path.join(dstdir, root[srcdir_prefix:], dirname)
            try:
                os.mkdir(dirpath)
            except OSError as e:
                if onerror is not None:
                    onerror(e)
