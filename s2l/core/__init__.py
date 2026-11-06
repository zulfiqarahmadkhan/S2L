"""Core segmentation, ROI conversion, and summary generation.

Heavy imports (torch, cellpose, stardist) are deferred to avoid
crashing at startup in PyInstaller builds.
"""
