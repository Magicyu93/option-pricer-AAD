"""One place to turn loose numbers into tensors without breaking the graph."""

from __future__ import annotations

import torch
from torch import Tensor

#: Guards logs and divisions against exactly-zero inputs. Small enough not to
#: perturb any price a desk would quote, large enough to keep float64 finite.
EPS = 1e-12


def as_tensor(x, dtype: torch.dtype | None = None, device=None) -> Tensor:
    """Coerce ``x`` to a tensor, passing existing tensors through untouched.

    Passing tensors through by identity is the point: a tensor that reaches here
    may be an ``nn.Parameter`` the caller intends to differentiate against, and
    rebuilding it would silently cut it out of the autograd graph.
    """
    if isinstance(x, Tensor):
        if dtype is not None and x.dtype != dtype:
            x = x.to(dtype)
        if device is not None and x.device != torch.device(device):
            x = x.to(device)
        return x
    return torch.as_tensor(x, dtype=dtype or torch.get_default_dtype(), device=device)


def interp_linear(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """Linear interpolation of ``fp`` over knots ``xp``, evaluated at ``x``.

    ``xp`` must be sorted and hold at least two points. Values outside the knot
    range are extrapolated along the nearest segment; callers that want another
    rule outside the knots clamp ``x`` first and patch the ends. Differentiable
    in ``x`` and ``fp``; the knot lookup itself is not, and need not be.
    """
    idx = torch.searchsorted(xp, x.detach().contiguous()).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    return f0 + (f1 - f0) * (x - x0) / (x1 - x0)


def hermite_slopes(xp: Tensor, fp: Tensor, start: Tensor, end: Tensor) -> Tensor:
    """Knot slopes for a monotone ``C^1`` cubic through increasing ``fp``.

    Steffen's (1990) interior slopes, ``min(2 s_left, 2 s_right, p)`` with
    ``p`` the length-weighted mean of the neighbouring secants, keep every
    segment monotone. The end slopes are the caller's -- typically the slope of
    whatever extrapolation continues the curve, so the joins are ``C^1`` too --
    capped at three times the end secant, Fritsch & Carlson's bound beyond which
    a cubic overshoots. Where that cap binds the join is only ``C^0``.
    """
    h = torch.diff(xp)
    s = torch.diff(fp) / h
    if s.numel() == 1:
        inner = s.new_zeros(0)
    else:
        p = (s[:-1] * h[1:] + s[1:] * h[:-1]) / (h[:-1] + h[1:])
        inner = torch.minimum(torch.minimum(2 * s[:-1], 2 * s[1:]), p)
    first = torch.minimum(start.reshape(1), 3 * s[:1])
    last = torch.minimum(end.reshape(1), 3 * s[-1:])
    return torch.cat([first, inner, last])


def interp_hermite(x: Tensor, xp: Tensor, fp: Tensor, slopes: Tensor) -> Tensor:
    """Cubic Hermite interpolation of ``fp`` with knot ``slopes``, evaluated at ``x``.

    Same knot rules as :func:`interp_linear`: ``xp`` sorted, at least two
    points, and callers clamp ``x`` to choose their own rule outside the knots.
    """
    idx = torch.searchsorted(xp, x.detach().contiguous()).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    h = x1 - x0
    u = (x - x0) / h
    u2, u3 = u * u, u * u * u
    return (
        (2 * u3 - 3 * u2 + 1) * fp[idx - 1]
        + (u3 - 2 * u2 + u) * h * slopes[idx - 1]
        + (-2 * u3 + 3 * u2) * fp[idx]
        + (u3 - u2) * h * slopes[idx]
    )
