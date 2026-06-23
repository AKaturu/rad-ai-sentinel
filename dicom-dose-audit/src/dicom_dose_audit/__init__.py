"""dicom-dose-audit: open-source CT radiation-dose audit tool.

Read CTDIvol and DLP from DICOM image headers or CT Radiation Dose Structured
Reports (RDSR), group studies by protocol / scanner / patient-size category,
detect missing dose metadata, flag statistical outliers, compare protocol
versions, plot monthly trends, and generate quality-improvement reports.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
