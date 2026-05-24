class NSAError(RuntimeError):
    """Base class for package-specific runtime failures."""


class ConfigError(NSAError):
    """Raised when a YAML config is missing required Phase 0 settings."""


class DatasetAccessError(NSAError):
    """Raised when a configured prompt dataset cannot be loaded."""


class LayerPathError(NSAError):
    """Raised when the requested model layer path is unavailable."""


class HookFailure(NSAError):
    """Raised when the activation hook cannot capture or mutate tensors."""


class ActivationMismatchError(NSAError):
    """Raised when observed activation statistics do not match the reference."""


class MissingDependencyError(NSAError):
    """Raised when an optional runtime dependency is unavailable."""


class UnsupportedBackendError(NSAError):
    """Raised when a backend is configured but intentionally not implemented."""
