"""SEC acquisition with an identified client, caching, throttling and bounded retries."""

import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from policypilot.models import Filing


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "ix:header"]):
        tag.decompose()
    return "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())


class SECClient:
    def __init__(self, user_agent: str, cache: Path, session=None, sleep=time.sleep):
        if not re.search(r"[^\s@]+@[^\s@]+\.[^\s@]+", user_agent) or "example.com" in user_agent:
            raise ValueError("Set SEC_USER_AGENT to PolicyPilot YourName your real contact email")
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.sleep = sleep

    def get(self, url: str, immutable: bool = False) -> str:
        key = self.cache / (hashlib.sha256(url.encode()).hexdigest() + ".txt")
        if immutable and key.exists():
            return key.read_text()
        for attempt in range(4):
            self.sleep(0.25)  # At most four sequential requests per second.
            try:
                response = self.session.get(url, timeout=45)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < 3:
                        retry = response.headers.get("Retry-After", "")
                        self.sleep(min(float(retry), 60) if retry.isdigit() else 2**attempt)
                        continue
                response.raise_for_status()
                if immutable:
                    key.write_text(response.text)
                return response.text
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 3:
                    raise
                self.sleep(2**attempt)
        raise RuntimeError("SEC retry limit reached")

    def latest(self, ticker: str) -> Filing:
        companies = json.loads(self.get("https://www.sec.gov/files/company_tickers.json"))
        company = next((c for c in companies.values() if c["ticker"] == ticker.upper()), None)
        if company is None:
            raise ValueError(f"Unknown ticker: {ticker}")
        cik = int(company["cik_str"])
        data = json.loads(self.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))
        blocks = [data["filings"]["recent"]]
        # Usually the latest annual filing is in recent. Read older blocks only if needed.
        if "10-K" not in blocks[0]["form"]:
            for older in data["filings"].get("files", []):
                blocks.append(
                    json.loads(
                        self.get(
                            "https://data.sec.gov/submissions/" + quote(older["name"], safe="")
                        )
                    )
                )
        candidates = []
        for block in blocks:
            for i, form in enumerate(block["form"]):
                if form == "10-K":
                    accession = block["accessionNumber"][i]
                    doc = quote(block["primaryDocument"][i], safe="")
                    candidates.append(
                        Filing(
                            ticker=ticker.upper(),
                            cik=cik,
                            company=data["name"],
                            accession=accession,
                            filing_date=block["filingDate"][i],
                            report_date=block["reportDate"][i],
                            source_url=f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                            f"{accession.replace('-', '')}/{doc}",
                        )
                    )
        if not candidates:
            raise ValueError(f"No 10-K found for {ticker}")
        return max(candidates, key=lambda f: (f.filing_date, f.accession))

    def download(self, filing: Filing) -> str:
        return self.get(filing.source_url, immutable=True)
