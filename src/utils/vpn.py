"""VPN connectivity pre-flight check. Called at bot startup."""

import json
import logging
import urllib.request

log = logging.getLogger(__name__)


def check_vpn(abort_if_us: bool = True) -> bool:
    """Return True if VPN appears active (non-US IP). Raise or warn if US-based."""
    try:
        with urllib.request.urlopen("https://ipinfo.io/json", timeout=5) as resp:
            info = json.loads(resp.read())
        country = info.get("country", "??")
        ip = info.get("ip", "??")
        log.info(f"[VPN] External IP: {ip}, Country: {country}")
        if country == "US":
            msg = f"[VPN] IP appears US-based ({ip}). VPN not connected or not routing correctly."
            if abort_if_us:
                raise RuntimeError(msg)
            log.warning(msg)
            return False
        return True
    except RuntimeError:
        raise
    except Exception as e:
        log.warning(f"[VPN] Could not verify IP location: {e}")
        return False
