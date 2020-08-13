import pyopencl as cl
import pyopencl.array as pycl_array

# TODO: make this a Singleton class


class ClObject:

    def __init__(self, queue, context):
        self.context = context
        self.queue = queue


class Code(ClObject):
    def __init__(self, queue, context):
        ClObject.__init__(self, queue, context)
        self.code = None
        self.program = None

    def build(self):
        if self.code is None:
            raise ValueError("There is no code defined")
        if self.program is None:
            self.program = cl.Program(self.context, self.code).build()

    def set_code(self, path_to_file=None, code=None):
        if path_to_file is None and code is None:
            raise ValueError("Invalid arguments")
        if path_to_file is None:
            self.code = code
        else:
            with open(path_to_file, 'r') as f:
                self.code = f.read()
