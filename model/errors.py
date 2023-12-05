class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class PropagationTypeError(Error):
    """Exception raised for errors in the input.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self):
        self.message = "propagation type not supported"
