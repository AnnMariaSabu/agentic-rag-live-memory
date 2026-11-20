"""Currency tool: live exchange rates from Frankfurter (European Central Bank data).

The agent (Step 4) will call get_exchange_rate(). It NEVER raises:
on success it returns rate info, on failure {"error": "<friendly message>"},
so the LLM can explain the problem to the user instead of crashing.
"""

from datetime import datetime, timezone

import requests
from pydantic import BaseModel, Field

import config

SOURCE = "Frankfurter (ECB)"


class ExchangeRate(BaseModel):
    """Shape of a successful answer. Pydantic checks it for us."""

    from_currency: str = Field(serialization_alias="from")
    to_currency: str = Field(serialization_alias="to")
    rate: float = Field(gt=0)  # a rate must be positive
    date: str  # day the rate was published
    source: str
    as_of: str  # when WE fetched it (UTC)
    amount: float | None = None  # optional: how much money to convert
    converted: float | None = None  # amount * rate, computed by code (LLMs are bad at maths)


def _clean_amount(amount):
    """None stays None; '220' or 220 -> 220.0; anything else -> 'bad'."""
    if amount is None:
        return None
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return "bad"
    return value if value > 0 else "bad"


def _result(src, dst, rate, date, amount) -> dict:
    result = ExchangeRate(
        from_currency=src, to_currency=dst, rate=rate, date=date,
        source=SOURCE, as_of=_now(), amount=amount,
        converted=round(amount * rate, 2) if amount is not None else None,
    )
    return result.model_dump(by_alias=True, exclude_none=True)


def _clean_code(code) -> str | None:
    """' usd' -> 'USD'. Returns None if it is not 3 letters."""
    if not isinstance(code, str):
        return None
    code = code.strip().upper()
    return code if len(code) == 3 and code.isalpha() else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_exchange_rate(from_currency: str, to_currency: str, amount: float | None = None) -> dict:
    """How much 1 unit of from_currency is worth in to_currency.
    With amount, also returns 'converted' = amount in to_currency."""
    src, dst = _clean_code(from_currency), _clean_code(to_currency)
    if src is None or dst is None:
        bad = from_currency if src is None else to_currency
        return {"error": f"'{bad}' is not a valid currency code. Use 3 letters, e.g. USD, EUR, JPY."}
    amount = _clean_amount(amount)
    if amount == "bad":
        return {"error": "The amount must be a positive number, e.g. 220."}

    # Same currency: answer ourselves (the API rejects this pair with 422).
    if src == dst:
        return _result(src, dst, 1.0, _now()[:10], amount)

    try:
        r = requests.get(
            f"{config.FRANKFURTER_URL}/latest",
            params={"from": src, "to": dst},
            timeout=config.TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        return _result(src, dst, data["rates"][dst], data["date"], amount)

    except requests.Timeout:
        return {"error": "The currency service took too long to answer. Please try again later."}
    except requests.ConnectionError:
        return {"error": "Could not reach the currency service. Check the internet connection."}
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return {"error": f"Unknown currency code '{src}' or '{dst}'. Use codes like USD, EUR, JPY."}
        return {"error": f"The currency service returned an error ({e}). Please try again later."}
    except (KeyError, ValueError) as e:  # unexpected reply shape / failed validation
        return {"error": f"The currency service sent an unexpected answer ({type(e).__name__})."}


if __name__ == "__main__":
    # Quick demo: python -m tools.currency
    for a, b in [("USD", "JPY"), ("xyz", "EUR"), ("usd", "usd"), ("US", "JPY")]:
        print(f"{a} -> {b}: {get_exchange_rate(a, b)}")
