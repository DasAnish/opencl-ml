import pyopencl as cl
import pyopencl.array as pycl_array


class ClObject:

    def __init__(self, context, queue):
        self.context = context
        self.queue = queue

    def build(self, code):
        return cl.Program(self.context, code).build()