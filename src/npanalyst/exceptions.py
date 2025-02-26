class NpAnalystBaseException(Exception):
    pass


class InvalidFormatError(NpAnalystBaseException):
    pass


class InvalidErrorType(NpAnalystBaseException):
    pass


class MismatchedData(NpAnalystBaseException):
    pass
