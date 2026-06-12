"""
delta_client.py
---------------
Authenticated client for the Delta Exchange REST API.
Handles:
  - HMAC-SHA256 request signing
  - Fetching account balances and positions
  - Placing / cancelling orders
  - Querying order book for current price
"""

import hashlib
import hmac
import time
import json
import requests
from typing import Optional


LIVE_URL    = "https://api.delta.exchange"
TESTNET_URL = "https://testnet-api.delta.exchange"


class DeltaClient:
    def __init__(self, api_key: str, api_secret: str, mode: str = "testnet"):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.base_url   = TESTNET_URL if mode == "testnet" else LIVE_URL
        self.mode       = mode
        self.session    = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------
    # Request signing
    # ------------------------------------------------------------------

    def _sign(self, method: str, path: str, query: str = "", body: str = "") -> dict:
        """
        Delta Exchange HMAC-SHA256 signature.
        Signature input: method + timestamp + path + query_string + body
        """
        timestamp = str(int(time.time()))
        message   = method.upper() + timestamp + path + query + body
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return {
            "api-key":   self.api_key,
            "timestamp": timestamp,
            "signature": signature
        }

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        query  = ""
        if params:
            query = "?" + "&".join(f"{k}={v}" for k, v in params.items())
        headers = self._sign("GET", path, query)
        resp    = self.session.get(self.base_url + path + query, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        body_str = json.dumps(body, separators=(",", ":"))
        headers  = self._sign("POST", path, body=body_str)
        resp     = self.session.post(self.base_url + path, headers=headers, data=body_str, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str, body: dict = None) -> dict:
        body_str = json.dumps(body or {}, separators=(",", ":"))
        headers  = self._sign("DELETE", path, body=body_str)
        resp     = self.session.delete(self.base_url + path, headers=headers, data=body_str, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Market data (public, no auth needed)
    # ------------------------------------------------------------------

    def get_ticker(self, symbol: str) -> dict:
        """Returns bid, ask, last_price for a symbol."""
        resp = self.session.get(f"{self.base_url}/v2/tickers/{symbol}", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result", {})

    def get_last_price(self, symbol: str) -> float:
        ticker = self.get_ticker(symbol)
        return float(ticker.get("close", ticker.get("last_price", 0)))

    def get_products(self) -> list:
        """List all available products (to find product_id for a symbol)."""
        resp = self.session.get(f"{self.base_url}/v2/products", timeout=10)
        resp.raise_for_status()
        return resp.json().get("result", [])

    def get_product_id(self, symbol: str) -> Optional[int]:
        """Resolve symbol string to Delta product_id."""
        products = self.get_products()
        for p in products:
            if p.get("symbol") == symbol:
                return p["id"]
        return None

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_wallet(self) -> dict:
        data = self._get("/v2/wallet/balances")
        return data.get("result", {})

    def get_positions(self) -> list:
        data = self._get("/v2/positions/margined")
        return data.get("result", [])

    def get_open_orders(self, symbol: str = None) -> list:
        params = {"state": "open"}
        if symbol:
            params["product_symbol"] = symbol
        data = self._get("/v2/orders", params)
        return data.get("result", [])

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def place_order(
        self,
        product_id: int,
        side: str,           # "buy" or "sell"
        size: int,           # contracts
        order_type: str = "market_order",
        limit_price: float = None,
        stop_loss: float = None,
        take_profit: float = None,
        reduce_only: bool = False
    ) -> dict:
        """
        Place a market or limit order with optional bracket (SL/TP).

        Args:
            product_id:  Numeric product ID from get_product_id()
            side:        'buy' or 'sell'
            size:        Number of contracts
            order_type:  'market_order' or 'limit_order'
            limit_price: Required for limit orders
            stop_loss:   Optional stop-loss price
            take_profit: Optional take-profit price
            reduce_only: If True, only closes existing position
        """
        body = {
            "product_id": product_id,
            "side":       side,
            "size":       size,
            "order_type": order_type,
            "reduce_only": reduce_only
        }

        if order_type == "limit_order" and limit_price:
            body["limit_price"] = str(limit_price)

        # Bracket orders (stop loss + take profit)
        bracket_orders = []
        if stop_loss:
            bracket_orders.append({
                "order_type":   "stop_loss_order",
                "stop_price":   str(stop_loss),
                "side":         "sell" if side == "buy" else "buy",
                "size":         size
            })
        if take_profit:
            bracket_orders.append({
                "order_type":   "take_profit_order",
                "stop_price":   str(take_profit),
                "side":         "sell" if side == "buy" else "buy",
                "size":         size
            })

        if bracket_orders:
            body["bracket_orders"] = bracket_orders

        resp = self._post("/v2/orders", body)
        if not resp.get("success"):
            raise RuntimeError(f"Order failed: {resp}")
        return resp.get("result", {})

    def cancel_order(self, order_id: int, product_id: int) -> dict:
        body = {"id": order_id, "product_id": product_id}
        resp = self._delete("/v2/orders", body)
        return resp.get("result", {})

    def cancel_all_orders(self, product_id: int) -> dict:
        body = {"product_id": product_id, "cancel_limit_orders": True, "cancel_stop_orders": True}
        resp = self._delete("/v2/orders/all", body)
        return resp.get("result", {})

    def close_position(self, product_id: int) -> dict:
        """Close all open contracts for a product with a market reduce-only order."""
        positions = self.get_positions()
        for pos in positions:
            if pos.get("product_id") == product_id:
                size = abs(int(pos.get("size", 0)))
                if size == 0:
                    return {}
                side = "sell" if int(pos["size"]) > 0 else "buy"
                return self.place_order(product_id, side, size, reduce_only=True)
        return {}
