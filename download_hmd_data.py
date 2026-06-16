"""
Download HMD country data and save as parquet files in data/.

Run once: python download_hmd_data.py
You will be prompted for your HMD email and password.
Files saved: data/{CODE}_{sex}.parquet  (Age x Year DataFrame of central death rates mx)
"""
import os, ssl, re, getpass, io
import http.cookiejar, urllib.parse, urllib.request
import numpy as np
import pandas as pd

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


def _hmd_login(username, password):
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=_SSL_CTX),
        urllib.request.HTTPCookieProcessor(jar),
    )
    login_url = "https://mortality.org/Account/Login"
    r = opener.open(login_url, timeout=20)
    html = r.read().decode("utf-8", errors="replace")
    m = re.search(r'__RequestVerificationToken[^>]*value=["\']([^"\']+)', html)
    csrf = m.group(1) if m else ""
    payload = urllib.parse.urlencode({
        "__RequestVerificationToken": csrf,
        "Email": username,
        "Password": password,
    }).encode()
    req = urllib.request.Request(
        login_url, data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Referer": login_url},
    )
    opener.open(req, timeout=20)
    return opener


def _fetch_lt_file(opener, url):
    with opener.open(url, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    if "<title>Login</title>" in raw or "<!DOCTYPE html>" in raw[:200]:
        raise RuntimeError("Authentication failed — check credentials")
    lines = [l for l in raw.splitlines() if l.strip() and not l.startswith("#")]
    header_idx = next(i for i, l in enumerate(lines) if "Year" in l and "Age" in l)
    col_map = {c.lower(): i for i, c in enumerate(lines[header_idx].split())}
    yi = col_map.get("year", 0)
    ai = col_map.get("age", 1)
    mi = col_map.get("mx", 2)
    rows = []
    for line in lines[header_idx + 1:]:
        parts = line.split()
        if len(parts) <= max(yi, ai, mi):
            continue
        try:
            year = int(parts[yi])
            age  = int(parts[ai].replace("+", ""))
            mx   = float(parts[mi]) if parts[mi] != "." else np.nan
            rows.append((year, age, mx))
        except ValueError:
            continue
    df = pd.DataFrame(rows, columns=["Year", "Age", "mx"])
    df["mx"].replace(0, 1e-6, inplace=True)
    df["mx"].fillna(1e-6, inplace=True)
    return df.pivot(index="Age", columns="Year", values="mx")


COUNTRIES = [
    {"name": "Australia",       "code": "AUS",     "start_long": 1921},
    {"name": "Canada",          "code": "CAN",     "start_long": 1921},
    {"name": "Denmark",         "code": "DNK",     "start_long": 1900},
    {"name": "England & Wales", "code": "GBRTENW", "start_long": 1922},
    {"name": "Finland",         "code": "FIN",     "start_long": 1900},
    {"name": "France",          "code": "FRATNP",  "start_long": 1900},
    {"name": "Italy",           "code": "ITA",     "start_long": 1900},
    {"name": "Norway",          "code": "NOR",     "start_long": 1900},
    {"name": "Sweden",          "code": "SWE",     "start_long": 1900},
    {"name": "Switzerland",     "code": "CHE",     "start_long": 1900},
]
SEX_FILES = {"male": "mltper_1x1.txt", "female": "fltper_1x1.txt"}

if __name__ == "__main__":
    print("HMD Data Download Script")
    print("=" * 40)
    username = input("HMD email: ")
    password = getpass.getpass("HMD password: ")

    print("\nLogging in ...", end=" ")
    try:
        opener = _hmd_login(username, password)
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        raise SystemExit(1)

    os.makedirs("data", exist_ok=True)
    ok, failed = 0, []

    for c in COUNTRIES:
        for sex, fname in SEX_FILES.items():
            out_path = os.path.join("data", f"{c['code']}_{sex}.parquet")
            if os.path.exists(out_path):
                print(f"  {c['code']} {sex}: already exists, skipping")
                ok += 1
                continue
            url = f"https://mortality.org/File/GetDocument/hmd.v6/{c['code']}/STATS/{fname}"
            print(f"  {c['code']:10s} {sex} ... ", end="", flush=True)
            try:
                df = _fetch_lt_file(opener, url)
                df.to_parquet(out_path)
                print(f"saved  ages {df.index[0]}-{df.index[-1]}, "
                      f"years {df.columns[0]}-{df.columns[-1]}")
                ok += 1
            except Exception as e:
                print(f"FAILED: {e}")
                failed.append(f"{c['code']} {sex}: {e}")

    print(f"\nDone. {ok} files saved.")
    if failed:
        print("Failed:")
        for f in failed:
            print(f"  {f}")
