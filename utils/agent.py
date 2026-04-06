import json
import ssl
import threading

import requests
import websocket


class Agent:
    """Client for the Ikigai Agent/Chat API.

    Args:
        user_email: Ikigai account email.
        api_key: Ikigai API key.
        base_url: API base URL (default ``https://api.ikigailabs.io``).
        app: Optional ikigai ``App`` object. When provided, ``project_id``
             and dataset-name resolution are handled automatically.
    """

    def __init__(self, user_email, api_key, base_url="https://api.ikigailabs.io", app=None):
        self._base_url = str(base_url).rstrip("/")
        self._headers = {"User": user_email, "Api-key": api_key}
        self._app = app

    def _request(self, method, endpoint, params=None, json_data=None):
        url = f"{self._base_url}/component/{endpoint}"
        resp = requests.request(
            method, url,
            headers=self._headers,
            params=params,
            json=json_data,
            verify=False,
        )
        resp.raise_for_status()
        return resp.json()

    def _resolve_project_id(self, project=None, project_id=None):
        """Return ``(project_id_str, app_or_none)``."""
        if project is not None:
            return project.app_id, project
        if project_id is not None:
            return project_id, self._app
        if self._app is not None:
            return self._app.app_id, self._app
        raise ValueError(
            "No project_id available. Provide project, project_id, "
            "or initialize Agent with an app."
        )

    # ------------------------------------------------------------------
    # Dataset ↔ Agent access
    # ------------------------------------------------------------------

    def _serialize_data_types(self, dataset):
        """Convert a PCL Dataset's ``data_types`` to the API dict format."""
        return {
            col: dt.model_dump(mode="json")
            for col, dt in dataset.data_types.items()
        }

    def _set_agent_access(self, dataset, project_id, agent_access=True):
        """Send an ``edit-dataset`` request to toggle ``agent_access``.

        *dataset* is an ikigai PCL ``Dataset`` object.  The current
        ``directory`` is fetched via ``dataset.describe()`` so the
        dataset is not accidentally moved.
        """
        info = dataset.describe()
        payload = {
            "dataset": {
                "dataset_id": dataset.dataset_id,
                "name": dataset.name,
                "project_id": project_id,
                "directory": info["directory"],
                "data_types": self._serialize_data_types(dataset),
                "agent_access": agent_access,
            }
        }
        return self._request("POST", "edit-dataset", json_data=payload)

    # ------------------------------------------------------------------
    # Chat management
    # ------------------------------------------------------------------

    def create_chat(self, project=None, project_id=None, agent_type="GENERAL_AGENT"):
        """Create a new chat session.

        ``project`` can be an ikigai ``App`` object (``project_id`` is
        extracted automatically).  If omitted, the ``app`` passed at
        ``Agent`` init time is used.
        """
        pid, app = self._resolve_project_id(project, project_id)
        data = self._request(
            "POST", "create-chat",
            json_data={"project_id": pid, "agent_type": agent_type},
        )
        return Chat(
            agent=self, project_id=pid, chat_id=data["chat_id"],
            agent_type=agent_type, app=app,
        )

    def get_chats(self, project=None, project_id=None):
        """List existing chats for a project."""
        pid, app = self._resolve_project_id(project, project_id)
        data = self._request("GET", "get-chats", params={"project_id": pid})
        return [
            Chat(
                agent=self, project_id=pid, chat_id=c["chat_id"],
                name=c.get("name", ""), agent_type=c.get("agent_type", ""),
                app=app,
            )
            for c in data.get("chats", [])
        ]


class Chat:
    """A single chat session with the Ikigai Agent.

    Created via :meth:`Agent.create_chat` or :meth:`Agent.get_chats`.
    """

    def __init__(self, agent, project_id, chat_id, name="", agent_type="", app=None):
        self._agent = agent
        self.project_id = project_id
        self.chat_id = chat_id
        self.name = name
        self.agent_type = agent_type
        self._app = app

    def __repr__(self):
        label = self.name or self.chat_id
        return f"Chat('{label}')"

    def ask(self, query, datasets=None, dataset_ids=None, app=None,
            on_update=None, timeout=300):
        """Send a query and stream real-time updates via WebSocket.

        Args:
            query: The question to ask the agent.
            datasets: Optional ``{role: value}`` dict where *role* is a
                string like ``"primary"`` and *value* is one of:

                * An ikigai ``Dataset`` object (``dataset_id`` extracted).
                * A dataset **name** (resolved via the ``App`` object).
            dataset_ids: Optional ``{dataset_id: role}`` dict of
                pre-resolved IDs in the raw API format.  Entries here
                bypass name resolution entirely.
            app: Optional ``App`` for dataset-name resolution.  Falls back
                to the ``App`` set on this Chat or on the parent Agent.
            on_update: Optional ``fn(message_dict)`` callback invoked for
                every WebSocket status update.  When *None*, updates are
                printed to stdout.
            timeout: Max seconds to wait for a response (default 300).

        Returns:
            The agent's final response text.
        """
        query_dataset_ids = self._resolve_datasets(datasets, dataset_ids, app)
        self._ensure_agent_access(datasets, dataset_ids, app)
        ws_url = self._get_ws_url()

        result: dict[str, str | None] = {"response": None, "error": None}
        connected = threading.Event()
        done = threading.Event()

        def _on_open(ws):
            connected.set()

        def _on_message(ws, raw):
            data = json.loads(raw)
            if data.get("component_id") != self.chat_id:
                return
            if data.get("job_type") != "CREATE_EXCHANGE":
                return

            status = data.get("status", "")
            message = data.get("message", "")

            if on_update:
                on_update(data)
            elif status != "RENAMED_CHAT":
                print(f"[{status}] {message}")

            if status == "SUCCESS":
                result["response"] = message
                done.set()
                ws.close()
            elif status in ("FAILED", "ERROR"):
                result["error"] = message
                done.set()
                ws.close()

        def _on_error(ws, error):
            result["error"] = str(error)
            done.set()

        def _on_close(ws, *_args):
            done.set()

        ws_app = websocket.WebSocketApp(
            ws_url,
            header=dict(self._agent._headers),
            on_open=_on_open,
            on_message=_on_message,
            on_error=_on_error,
            on_close=_on_close,
        )

        ws_thread = threading.Thread(
            target=ws_app.run_forever,
            kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}},
            daemon=True,
        )
        ws_thread.start()

        if not connected.wait(timeout=10):
            ws_app.close()
            raise ConnectionError("WebSocket connection timed out")

        self._create_exchange(query, query_dataset_ids)

        done.wait(timeout=timeout)

        if result["error"]:
            raise RuntimeError(f"Agent query failed: {result['error']}")
        if result["response"] is None:
            raise TimeoutError("Agent did not respond within the timeout period")

        return result["response"]

    def get_exchanges(self):
        """Retrieve all exchanges (Q&A pairs) for this chat."""
        data = self._agent._request(
            "GET", "get-exchanges-for-chat",
            params={"project_id": self.project_id, "chat_id": self.chat_id},
        )
        return data.get("exchanges", [])

    def is_processing(self):
        """Check whether the agent is currently processing a query."""
        data = self._agent._request(
            "GET", "is-query-processing",
            params={"chat_id": self.chat_id},
        )
        return data.get("is_running", False)

    # -- internal helpers --------------------------------------------------

    def _get_ws_url(self):
        data = self._agent._request(
            "GET", "get-chat-websocket-url",
            params={"chat_id": self.chat_id, "project_id": self.project_id},
        )
        url = data.get("url", "")
        if not url:
            raise RuntimeError("Failed to obtain WebSocket URL")
        return url

    def _create_exchange(self, query, query_dataset_ids):
        return self._agent._request(
            "POST", "create-exchange",
            json_data={
                "project_id": self.project_id,
                "chat_id": self.chat_id,
                "query": query,
                "query_dataset_ids": query_dataset_ids,
            },
        )

    def _ensure_agent_access(self, datasets=None, dataset_ids=None, app=None):
        """Best-effort enable agent access for referenced datasets.

        Uses the PCL ``app.datasets[name]`` to look up each dataset
        and calls ``edit-dataset`` to grant access.  Failures are
        silently ignored so the query is never blocked.
        """
        resolved_app = app or self._app or self._agent._app
        if resolved_app is None:
            return

        # datasets = {role: name_or_object}
        for role, ds in (datasets or {}).items():
            try:
                pcl_ds = ds if hasattr(ds, "dataset_id") else resolved_app.datasets[ds]
                self._agent._set_agent_access(pcl_ds, self.project_id)
            except Exception:
                pass  # best-effort

        # dataset_ids = {dataset_id: role} — less common; find by ID
        if dataset_ids:
            # Build a quick id→Dataset lookup from the app
            try:
                id_map = {
                    resolved_app.datasets[n].dataset_id: resolved_app.datasets[n]
                    for n in resolved_app.datasets
                }
            except Exception:
                return
            for ds_id in dataset_ids:
                target = id_map.get(ds_id)
                if target is None:
                    continue
                try:
                    self._agent._set_agent_access(target, self.project_id)
                except Exception:
                    pass  # best-effort

    def _resolve_datasets(self, datasets=None, dataset_ids=None, app=None):
        query_dataset_ids: dict[str, str] = {}

        if dataset_ids:
            query_dataset_ids.update(dataset_ids)

        if not datasets:
            return query_dataset_ids

        resolved_app = app or self._app or self._agent._app

        for role, ds in datasets.items():
            if hasattr(ds, "dataset_id"):
                query_dataset_ids[ds.dataset_id] = role
            elif resolved_app is not None:
                try:
                    resolved = resolved_app.datasets[ds]
                    query_dataset_ids[resolved.dataset_id] = role
                except Exception:
                    query_dataset_ids[ds] = role
            else:
                query_dataset_ids[ds] = role

        return query_dataset_ids
