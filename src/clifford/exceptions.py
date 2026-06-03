class CliffordError(Exception):
    pass


class ModelNotFoundError(CliffordError):
    pass


class LayerError(CliffordError):
    pass


class ShapeMismatchError(CliffordError):
    pass


class OptimizerError(CliffordError):
    pass


class LossError(CliffordError):
    pass


class DatabaseError(CliffordError):
    pass


class ConfigurationError(CliffordError):
    pass


class TrainingError(CliffordError):
    pass


class DatasetError(CliffordError):
    pass


class PersistenceError(CliffordError):
    pass


class SearchError(CliffordError):
    pass
