import pyopencl as cl
import pyopencl.array as pycl_array

# TODO: make this a Singleton class


class ClSingleton:

    __instance = None

    @staticmethod
    def get_instance():
        if ClSingleton.__instance is None:
            ClSingleton()
        return ClSingleton.__instance

    def __init__(self):
        if ClSingleton.__instance is not None:
            raise Exception("There is already an instance please use the get_instance method.")
        else:
            ClSingleton.__instance = self

        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)


class Code:

    def __init__(self):
        self.cl = ClSingleton.get_instance()
        self.code = None
        self.program = None

    def build(self):
        if self.code is None:
            raise ValueError("There is no code defined")
        if self.program is None:
            self.program = cl.Program(self.cl.context, self.code).build()

    def set_code(self, path_to_file=None, code=None):
        if path_to_file is None and code is None:
            raise ValueError("Invalid arguments")
        if path_to_file is None:
            self.code = code
        else:
            with open(path_to_file, 'r') as f:
                self.code = f.read()
