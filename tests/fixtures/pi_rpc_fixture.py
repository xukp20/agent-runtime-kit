from __future__ import annotations

import json
import sys


for line in sys.stdin:
    request = json.loads(line)
    command = request.get("type")
    if command == "echo":
        print(json.dumps({"type": "message_start", "value": request.get("value")}), flush=True)
        print(
            json.dumps(
                {
                    "id": request["id"],
                    "type": "response",
                    "command": "echo",
                    "success": True,
                    "data": {"value": request.get("value")},
                }
            ),
            flush=True,
        )
    elif command == "fail":
        print(
            json.dumps(
                {
                    "id": request["id"],
                    "type": "response",
                    "command": "fail",
                    "success": False,
                    "error": "fixture failure",
                }
            ),
            flush=True,
        )
    elif command == "malformed":
        print("not-json", flush=True)
        break
    elif command == "exit":
        print("fixture stderr", file=sys.stderr, flush=True)
        raise SystemExit(7)
    elif command == "hang":
        continue
