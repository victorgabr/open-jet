"""NDJSON stdio server used by the bundled TypeScript TUI."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from .service_controller import OpenJetServiceController, ServiceError

ControllerError = ServiceError


PROTOCOL_VERSION = 1
REQUEST_TYPES = {
    "initialize",
    "command",
    "tool_execute",
    "generation_metrics",
    "agent_trace",
    "status",
    "resize",
    "shutdown",
    "confirm",
}


class ProtocolError(ControllerError):
    """Fatal framing or contract violation."""


class ProtocolServer:
    def __init__(self, *, force_setup: bool = False) -> None:
        self._write_lock = asyncio.Lock()
        self.controller = OpenJetServiceController(self.emit, confirm=self.confirm, force_setup=force_setup)
        self._running = True
        self._exit_code = 0
        self._confirm_waiters: dict[str, asyncio.Future[bool]] = {}
        self._confirm_seq = 0
        self._inflight: set[asyncio.Task] = set()

    async def emit(self, event_type: str, fields: dict[str, Any] | None = None) -> None:
        message = {"protocolVersion": PROTOCOL_VERSION, "type": event_type, **(fields or {})}
        encoded = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        async with self._write_lock:
            sys.stdout.write(encoded + "\n")
            sys.stdout.flush()

    async def error(self, text: str, *, request_id: str | None = None, fatal: bool = False) -> None:
        fields: dict[str, Any] = {"text": text, "payload": {"fatal": fatal}}
        if request_id:
            fields["requestId"] = request_id
        await self.emit("error", fields)

    async def confirm(self, question: str) -> bool:
        """Ask the frontend a yes/no question and wait for its `confirm` answer."""
        self._confirm_seq += 1
        waiter_id = f"confirm-{self._confirm_seq}"
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._confirm_waiters[waiter_id] = future
        try:
            await self.emit("confirm_request", {"id": waiter_id, "text": question})
            return await future
        finally:
            self._confirm_waiters.pop(waiter_id, None)

    async def handle(self, message: dict[str, Any]) -> None:
        request_type = str(message.get("type", ""))
        request_id = str(message.get("id", "")).strip()
        if message.get("protocolVersion") != PROTOCOL_VERSION:
            raise ProtocolError(
                f"Protocol mismatch: frontend={message.get('protocolVersion')!r}, backend={PROTOCOL_VERSION}."
            )
        if request_type not in REQUEST_TYPES:
            raise ProtocolError(f"Unknown request type: {request_type or '<missing>'}")
        if not request_id:
            raise ProtocolError("Every request requires a non-empty id.")
        try:
            self.controller.claim_request_id(request_id)
        except RuntimeError as exc:
            raise ProtocolError(str(exc)) from exc

        if request_type == "initialize":
            snapshot = await self.controller.initialize()
            await self.emit("ready", {"requestId": request_id, "payload": snapshot})
        elif request_type == "command":
            raw_key = message.get("apiKey")
            api_key = raw_key.strip() if isinstance(raw_key, str) else ""
            result = await self.controller.command(
                str(message.get("text", "")),
                api_key=api_key or None,
            )
            await self.emit("notification", {"requestId": request_id, **result})
        elif request_type == "tool_execute":
            payload = message.get("payload")
            if not isinstance(payload, dict):
                raise ProtocolError("tool_execute requires an object payload.")
            result = await self.controller.execute_openjet_tool(
                str(payload.get("name", "")),
                payload.get("arguments"),
                call_id=str(message.get("callId", "")).strip() or request_id,
            )
            await self.emit(
                "tool_result",
                {
                    "requestId": request_id,
                    "callId": result["callId"],
                    "text": result["output"],
                    "payload": result,
                },
            )
        elif request_type == "generation_metrics":
            payload = message.get("payload")
            if not isinstance(payload, dict):
                raise ProtocolError("generation_metrics requires an object payload.")
            self.controller.update_generation_metrics(payload)
        elif request_type == "agent_trace":
            payload = message.get("payload")
            if not isinstance(payload, dict):
                raise ProtocolError("agent_trace requires an object payload.")
            self.controller.record_agent_trace(payload)
        elif request_type == "status":
            await self.emit("state_snapshot", {"requestId": request_id, "payload": self.controller._snapshot()})
        elif request_type == "resize":
            self.controller.resize(int(message.get("width", 80)), int(message.get("height", 24)))
            await self.emit("status_update", {"requestId": request_id, "payload": {"resized": True}})
        elif request_type == "confirm":
            target = str(message.get("requestId", "")).strip()
            waiter = self._confirm_waiters.get(target)
            if waiter is None or waiter.done():
                raise ControllerError(f"No pending confirmation for request id: {target or '<missing>'}")
            waiter.set_result(bool(message.get("approved")))
        elif request_type == "shutdown":
            await self.controller.close()
            await self.emit("status_update", {"requestId": request_id, "payload": {"shutdown": True}})
            self._running = False

    async def run(self) -> int:
        while self._running:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                break
            request_id: str | None = None
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ProtocolError("Protocol message must be a JSON object.")
                request_id = str(value.get("id", "")).strip() or None
            except (ProtocolError, json.JSONDecodeError) as exc:
                await self.error(str(exc), request_id=request_id, fatal=True)
                return 2
            # Requests are dispatched concurrently: a command such as /setup can
            # await a frontend `confirm` answer that arrives through this same
            # stdin loop, so inline handling would deadlock. Correlation stays
            # per-message via request ids.
            task = asyncio.create_task(self._dispatch(value, request_id))
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)
        await self.controller.close()
        return self._exit_code

    async def _dispatch(self, message: dict[str, Any], request_id: str | None) -> None:
        try:
            await self.handle(message)
        except ProtocolError as exc:
            await self.error(str(exc), request_id=request_id, fatal=True)
            self._exit_code = 2
            self._running = False
        except (ControllerError, ValueError, TypeError) as exc:
            await self.error(str(exc), request_id=request_id)
        except Exception as exc:
            print(f"OpenJet backend failure: {exc}", file=sys.stderr, flush=True)
            await self.error(str(exc) or type(exc).__name__, request_id=request_id, fatal=True)
            self._exit_code = 1
            self._running = False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--force-setup", action="store_true")
    args = parser.parse_args(argv)
    return asyncio.run(ProtocolServer(force_setup=args.force_setup).run())


if __name__ == "__main__":
    raise SystemExit(main())
