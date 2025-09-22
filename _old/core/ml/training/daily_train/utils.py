import os, time, subprocess, datetime as dt, logging
import psycopg2
from psycopg2 import extensions as _pgext
from contextlib import contextmanager
from urllib.parse import urlsplit, urlunsplit, quote

def setup_logging(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

def set_thread_limits(n: int):
    """Limita hilos BLAS/NumPy para no pisar los procesos realtime."""
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[var] = str(max(1, int(n)))

def utc_today():
    # timezone-aware; evita DeprecationWarning de utcnow()
    return dt.datetime.now(dt.UTC).date()

def date_str(d: dt.date):
    return d.strftime("%Y-%m-%d")

def window_from_to(days: int, gap_days: int = 0):
    """from = hoy-(gap+days), to = hoy-gap (UTC-aware)"""
    to_d = dt.datetime.now(dt.UTC).date() - dt.timedelta(days=gap_days)
    from_d = to_d - dt.timedelta(days=days)
    return date_str(from_d), date_str(to_d)

def run_cmd(cmd: list[str]) -> int:
    logging.info("RUN: %s", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode

def _normalize_db_url(dsn: str) -> str:
    """
    Normaliza credenciales en URLs tipo postgresql://user:pass@host:port/db
    aplicando percent-encoding si hay tildes/espacios/símbolos.
    """
    try:
        u = urlsplit(dsn)
        if "@" not in u.netloc:
            return dsn
        userinfo, hostport = u.netloc.rsplit("@", 1)
        if ":" in userinfo:
            user, pwd = userinfo.split(":", 1)
            user_q = quote(user, safe="")
            pwd_q  = quote(pwd,  safe="")
            netloc = f"{user_q}:{pwd_q}@{hostport}"
        else:
            netloc = f"{quote(userinfo, safe='')}@{hostport}"
        return urlunsplit((u.scheme, netloc, u.path, u.query, u.fragment))
    except Exception:
        return dsn

@contextmanager
def pg_conn(dsn: str):
    """Context manager de conexión con normalización de URL."""
    safe_dsn = _normalize_db_url(dsn)

    def _connect_from_url(url: str):
        u = urlsplit(url)
        # userinfo
        user = u.username
        pwd  = u.password
        host = u.hostname or "localhost"
        port = u.port or 5432
        dbname = (u.path or "/").lstrip("/")
        # libpq extra params from query string (simple key=val&...)
        # psycopg2 accepts them in **kwargs if names match
        params = {}
        if u.query:
            for kv in u.query.split("&"):
                if not kv:
                    continue
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    params[k] = v
        conn_kw = {"dbname": dbname, "host": host, "port": port}
        if user is not None:
            conn_kw["user"] = user
        if pwd is not None:
            conn_kw["password"] = pwd
        # force client encoding via options
        if "options" in params:
            params["options"] += " -c client_encoding=utf8"
        else:
            params["options"] = "-c client_encoding=utf8"
        conn_kw.update(params)
        return psycopg2.connect(**conn_kw)

    try:
        if (safe_dsn.startswith("postgres://")
            or safe_dsn.startswith("postgresql://")
            or safe_dsn.startswith("postgresql+psycopg2://")):
            conn = _connect_from_url(safe_dsn)
        else:
            # DSN en formato key=value ... → parse manual y pasar kwargs
            kv = {}
            parts = [p for p in safe_dsn.strip().split() if p]
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    # quitar comillas si las hay
                    if v.startswith("'") and v.endswith("'"):
                        v = v[1:-1]
                    if v.startswith('"') and v.endswith('"'):
                        v = v[1:-1]
                    kv[k] = v
            if kv:
                # forzar client_encoding
                kv.setdefault("options", "-c client_encoding=utf8")
                conn = psycopg2.connect(**kv)
            else:
                conn = psycopg2.connect(safe_dsn)
    except UnicodeDecodeError:
        # Fallback: forzamos client_encoding y quoting completo
        try:
            u = _normalize_db_url(safe_dsn)
            if "?" in u:
                u += "&client_encoding=utf8"
            else:
                u += "?client_encoding=utf8"
            if u.startswith("postgres://") or u.startswith("postgresql://"):
                conn = _connect_from_url(u)
            else:
                # parse and pass kwargs instead of DSN string
                conn = _connect_from_url(u)
        except Exception as e:
            raise e
    conn.autocommit = True
    try:
        yield conn
    finally:
        conn.close()

def try_lock(conn, key: str) -> bool:
    """Advisory lock no bloqueante; true si obtiene el lock."""
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(hashtextextended(%s, 0));", (key,))
        row = cur.fetchone()
        return bool(row and row[0])

def release_lock(conn, key: str):
    """Libera el advisory lock si está tomado."""
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0));", (key,))
