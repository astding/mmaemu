"""Gated fast kernels v9 and tc3 (optional; not release-blocking).

Discovery: ``mmaemu.backends.cuda.fast.specs.SPECS`` (the v9 / tc3 KernelSpecs) and
``PROBE_MODULES`` ({"probes": "probes.cu"}); importing ``specs`` imports ``gates``, which registers the "window" and
"design_a" gates with ``mmaemu.backends.cuda.spec.register_gate``.

Both kernels have hardware-dependent numerics: they are auto-selected only on validated capabilities (today (8, 9))
or with MMAEMU_ALLOW_GATED_KERNELS=1, and only after their gates pass on the device.  See docs/kernels_fast.md.
"""
