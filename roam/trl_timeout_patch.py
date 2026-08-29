"""Inject a default per-request timeout into TRL's VLLMClient session.

Without this, TRL's requests.Session has no timeout, so a stale keep-alive
connection blocks forever. This patch makes dead sockets raise ReadTimeout,
letting TRL's existing Retry strategy reconnect.

Idempotent. Call apply() before any VLLMClient is constructed.
"""
import os

_APPLIED = False


def apply():
    global _APPLIED
    if _APPLIED:
        return
    try:
        import trl.generation.vllm_client as _vc
    except Exception as e:
        print(f"[trl_timeout_patch] could not import TRL vllm_client ({e!r}); NOT patched", flush=True)
        return

    connect_t = float(os.environ.get("TRL_VLLM_CONNECT_TIMEOUT", "10"))
    read_t = float(os.environ.get("TRL_VLLM_READ_TIMEOUT", "600"))
    _orig_init = _vc.VLLMClient.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        sess = getattr(self, "session", None)
        if sess is None:
            return
        _orig_request = sess.request

        def _request_with_timeout(method, url, **kw):
            kw.setdefault("timeout", (connect_t, read_t))
            return _orig_request(method, url, **kw)

        sess.request = _request_with_timeout

    _vc.VLLMClient.__init__ = _patched_init
    _APPLIED = True
    print(f"[trl_timeout_patch] VLLMClient session timeout = (connect={connect_t}s, read={read_t}s)",
          flush=True)
