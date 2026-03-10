"""Bokeh-based interactive scene viewer for 123D.

Usage::

    from py123d.visualization.bokeh import BokehViewer, create_viewer_from_log_dirs
"""

__all__ = [
    "BokehViewer",
    "create_viewer_from_log_dirs",
    "create_viewer_from_scenes",
]


def __getattr__(name):
    if name in __all__:
        from py123d.visualization.bokeh.bokeh_viewer import (
            BokehViewer,
            create_viewer_from_log_dirs,
            create_viewer_from_scenes,
        )
        _exports = {
            "BokehViewer": BokehViewer,
            "create_viewer_from_log_dirs": create_viewer_from_log_dirs,
            "create_viewer_from_scenes": create_viewer_from_scenes,
        }
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
