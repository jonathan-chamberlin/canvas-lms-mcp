import re
from typing import Any, Dict, Optional

import httpx

_LINK_NEXT_RE = re.compile(r'<([^>]+)>\s*;\s*rel="next"')


def _next_link(link_header: Optional[str]) -> Optional[str]:
    """Extract the rel="next" URL from a Canvas Link header, if present."""
    if not link_header:
        return None
    match = _LINK_NEXT_RE.search(link_header)
    return match.group(1) if match else None


class CanvasClient:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CanvasClient, cls).__new__(cls)
        return cls._instance

    def __init__(
        self, api_token: str = None, base_url: str = "https://canvas.instructure.com"
    ):
        # Only initialize once
        if not self._initialized and api_token is not None:
            self.api_token = api_token
            self.base_url = base_url
            self.client = httpx.AsyncClient(
                base_url=base_url,
                headers={
                    "Authorization": f"Bearer {api_token}",
                    "Content-Type": "application/json",
                    "User-Agent": "canvas-lms-mcp/0.1.3",
                },
            )
            self._initialized = True

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise RuntimeError(
                "CanvasClient has not been initialized. Call CanvasClient(api_token) first."
            )
        return cls._instance

    async def get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Make a GET request to the Canvas API (single page)."""
        response = await self.client.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()

    async def get_all(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        max_pages: int = 50,
    ) -> list:
        """GET a list endpoint and follow Canvas Link-header pagination.

        Canvas returns a `Link` header with rel="next" on paginated list
        endpoints. Walk it to collect every page into a single list. Bounded
        by ``max_pages`` to avoid runaway loops on a misbehaving server.

        Assumes the endpoint returns a JSON array. Raises ValueError if the
        first response is not a list.
        """
        merged_params = dict(params or {})
        merged_params.setdefault("per_page", 100)

        response = await self.client.get(endpoint, params=merged_params)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(
                f"get_all expected a JSON array from {endpoint}, "
                f"got {type(payload).__name__}"
            )

        results: list = list(payload)
        pages = 1
        next_url = _next_link(response.headers.get("link"))
        while next_url and pages < max_pages:
            # Canvas next URLs are absolute; httpx handles absolute URLs fine.
            response = await self.client.get(next_url)
            response.raise_for_status()
            chunk = response.json()
            if not isinstance(chunk, list):
                break
            results.extend(chunk)
            pages += 1
            next_url = _next_link(response.headers.get("link"))
        return results

    async def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a POST request to the Canvas API."""
        response = await self.client.post(endpoint, json=data)
        response.raise_for_status()
        return response.json()

    async def put(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a PUT request to the Canvas API."""
        response = await self.client.put(endpoint, json=data)
        response.raise_for_status()
        return response.json()

    async def post_multipart(
        self, url: str, data: Dict[str, Any], file_path: str, file_field: str = "file"
    ) -> httpx.Response:
        """Upload a file via multipart form data to an arbitrary URL (no base_url prefix)."""
        import mimetypes
        import os

        filename = os.path.basename(file_path)
        content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"

        with open(file_path, "rb") as f:
            files = {file_field: (filename, f, content_type)}
            # Use a fresh client without the JSON content-type header
            async with httpx.AsyncClient() as upload_client:
                response = await upload_client.post(url, data=data, files=files)
                response.raise_for_status()
                return response

    async def upload_file_to_endpoint(
        self, endpoint: str, file_path: str, file_name: str
    ) -> Dict[str, Any]:
        """Canvas 3-step file upload: notify, upload bytes, confirm.

        Args:
            endpoint: Canvas API endpoint to initiate upload (e.g., /api/v1/groups/{id}/files)
            file_path: Local path to the file to upload
            file_name: Desired filename on Canvas

        Returns:
            File object dict from Canvas
        """
        import os

        file_size = os.path.getsize(file_path)

        # Step 1: Tell Canvas about the file
        step1_response = await self.post(endpoint, {
            "name": file_name,
            "size": file_size,
        })

        upload_url = step1_response["upload_url"]
        upload_params = step1_response.get("upload_params", {})

        # Step 2: POST the file to the upload URL
        step2_response = await self.post_multipart(
            upload_url, upload_params, file_path
        )

        # Step 3: Canvas may return the file directly (201) or a redirect (3xx)
        if step2_response.status_code in (200, 201):
            return step2_response.json()

        # Handle redirect confirmation
        if step2_response.status_code in (301, 302, 303):
            confirm_url = step2_response.headers["Location"]
            confirm_response = await self.client.get(confirm_url)
            confirm_response.raise_for_status()
            return confirm_response.json()

        return step2_response.json()

    async def delete(self, endpoint: str) -> Dict[str, Any]:
        """Make a DELETE request to the Canvas API."""
        response = await self.client.delete(endpoint)
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the HTTP client."""
        self.client.close()
