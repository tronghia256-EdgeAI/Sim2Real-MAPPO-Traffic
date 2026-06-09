from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import mimetypes
import uuid

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TelegramConfig:
    bot_token: str
    chat_id: str
    parse_mode: str = "HTML"
    disable_web_page_preview: bool = True
    timeout: float = 10.0
    enabled: bool = True


class TelegramAlertService:
    """
    Telegram notifier.

    Supported tele.json formats:
    1) Flat:
       {
         "bot_token": "123:ABC",
         "chat_id": "-1001234567890"
       }

    2) Nested:
       {
         "telegram": {
           "bot_token": "123:ABC",
           "chat_id": "-1001234567890",
           "parse_mode": "HTML"
         }
       }
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        parse_mode: str = "HTML",
        disable_web_page_preview: bool = True,
        timeout: float = 10.0,
        enabled: bool = True,
    ) -> None:
        self.config = TelegramConfig(
            bot_token=bot_token.strip(),
            chat_id=str(chat_id).strip(),
            parse_mode=parse_mode,
            disable_web_page_preview=bool(disable_web_page_preview),
            timeout=float(timeout),
            enabled=bool(enabled),
        )

    @classmethod
    def from_config(cls, config_path: str | Path) -> "TelegramAlertService":
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Telegram config not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        tg = cfg.get("telegram", cfg) if isinstance(cfg, dict) else {}
        bot_token = tg.get("bot_token") or tg.get("token") or tg.get("api_token")
        chat_id = tg.get("chat_id") or tg.get("chatid")
        parse_mode = tg.get("parse_mode", "HTML")
        disable_preview = bool(tg.get("disable_web_page_preview", True))
        timeout = float(tg.get("timeout", 10.0))
        enabled = bool(tg.get("enabled", True))

        if not bot_token or not chat_id:
            logger.warning("Telegram config missing bot_token/chat_id; alerts disabled.")
            return cls(
                bot_token="",
                chat_id="",
                parse_mode=parse_mode,
                disable_web_page_preview=disable_preview,
                timeout=timeout,
                enabled=False,
            )

        return cls(
            bot_token=str(bot_token),
            chat_id=str(chat_id),
            parse_mode=parse_mode,
            disable_web_page_preview=disable_preview,
            timeout=timeout,
            enabled=enabled,
        )

    @property
    def is_enabled(self) -> bool:
        return self.config.enabled and bool(self.config.bot_token) and bool(self.config.chat_id)

    def send_message(self, text: str) -> bool:
        if not self.is_enabled:
            logger.info("Telegram alert skipped: service disabled or config incomplete.")
            return False

        url = f"https://api.telegram.org/bot{self.config.bot_token}/sendMessage"
        data: Dict[str, Any] = {
            "chat_id": self.config.chat_id,
            "text": text,
            "disable_web_page_preview": str(self.config.disable_web_page_preview).lower(),
        }
        if self.config.parse_mode:
            data["parse_mode"] = self.config.parse_mode

        try:
            req = Request(
                url,
                data=urlencode(data).encode("utf-8"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            with urlopen(req, timeout=self.config.timeout) as resp:
                _ = resp.read()
            return True
        except (HTTPError, URLError, TimeoutError, OSError):
            logger.exception("Failed to send Telegram text alert")
            return False

    @staticmethod
    def _encode_image(frame: np.ndarray) -> bytes:
        if frame is None:
            raise ValueError("frame cannot be None")
        if frame.ndim != 3:
            raise ValueError("frame must be a color image (H, W, 3)")
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Failed to encode image as JPEG")
        return buf.tobytes()

    def send_photo(self, frame: np.ndarray, caption: str = "") -> bool:
        """
        Send a full-frame image to Telegram with optional caption.
        """
        if not self.is_enabled:
            logger.info("Telegram photo alert skipped: service disabled or config incomplete.")
            return False

        url = f"https://api.telegram.org/bot{self.config.bot_token}/sendPhoto"
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        image_bytes = self._encode_image(frame)

        fields = [
            ("chat_id", self.config.chat_id.encode("utf-8")),
            ("caption", caption.encode("utf-8")),
            ("disable_web_page_preview", str(self.config.disable_web_page_preview).lower().encode("utf-8")),
        ]
        if self.config.parse_mode:
            fields.append(("parse_mode", self.config.parse_mode.encode("utf-8")))

        body = bytearray()
        for name, value in fields:
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
            body.extend(value)
            body.extend(b"\r\n")

        filename = "accident.jpg"
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="photo"; filename="{filename}"\r\n'.encode("utf-8")
        )
        body.extend(b"Content-Type: image/jpeg\r\n\r\n")
        body.extend(image_bytes)
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))

        try:
            req = Request(
                url,
                data=bytes(body),
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
            with urlopen(req, timeout=self.config.timeout) as resp:
                _ = resp.read()
            return True
        except (HTTPError, URLError, TimeoutError, OSError):
            logger.exception("Failed to send Telegram photo alert")
            return False

    def send_accident_alert(self, text: str, frame: Optional[np.ndarray] = None) -> bool:
        """
        Send text-only or photo alert.
        If frame is provided, send photo with caption.
        """
        if frame is not None:
            return self.send_photo(frame, caption=text)
        return self.send_message(text)