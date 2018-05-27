class InvalidArgType(Exception):
    def __init__(self, arg, arg_index):
        self.message = "=> %s has invalid argument type at pos %d. Expected a list." % (str(arg), arg_index)
        super(InvalidArgType, self).__init__(self.message)


class InvalidArgCount(Exception):
    def __init__(self, arg):
        self.message = "Only " + str(len(arg)) + " provided, expecting at least 2 lists."
