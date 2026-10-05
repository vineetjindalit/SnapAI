"""GPU / device-picking utilities. Single source of truth for cuda/mps/cpu."""
from .device import get_device, device_summary, DEVICE  # noqa: F401
