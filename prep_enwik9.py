"""Bit-exact reversible preprocessor for enwik9 (Goal B: official Hutter Prize S1+S2 without cmix).

Losslessly separates enwik9 (1,000,000,000 bytes) into:
  1. enwik9.prep.meta.lzma  — Columnar XML metadata (page <id> deltas, rev <id>, rid-sorted
                              timestamp deltas, frequency-indexed contributor table, flags,
                              restrictions, comments, XML header & truncated final page).
  2. enwik9.prep            — Clean grouped text stream ([redirects] + [US census] + [articles]) with:
                              - Zero XML tag/metadata boilerplate (~90 MB removed)
                              - Bijective XML entity unescaping (&lt; &gt; &quot; &amp; -> < > " &, ~20 MB removed)
                              - Exact <title> (\x03) and short-title (\x04) back-references
                              - Piped-link target shorthand (\x07 short_prefix, \x08 short_lower, \x0b exact_lower)
                              - Zero-permutation redirect & US Census grouping (via 2 bits in flags)
                              - Exact/1-digit-delta km2 <-> mi2 unit conversion factoring
  3. enwik9.prep.meta.json  — Exact metadata byte overhead + n_bytes_prep + SHA-256 verification.

Every single byte of enwik9 is reconstructed and verified via SHA-256 against the original file.
"""
import calendar
import hashlib
import json
import lzma
import os
import re
import struct
import time
from collections import Counter
import numpy as np

# Unused control bytes in enwik9 (verified 0 occurrences across all 1,000,000,000 bytes):
B_ART = b"\x01"      # start of article: \x01<title>\x02<body>
B_SEP = b"\x02"      # separator between title and body
B_TITLE = b"\x03"    # exact <title> back-reference inside body
B_STITLE = b"\x04"   # short <title> (prefix before ", " or " (") back-reference inside body
B_RSTD = b"\x05"     # standard redirect "#REDIRECT [[target]]" -> \x05<title>\x02<target>
B_RRAW = b"\x06"     # non-standard redirect -> \x06<title>\x02<raw_body>
B_LP_SH = b"\x07"    # piped link [[Target|short(Target)]] -> [[Target\x07]]
B_LP_SHL = b"\x08"   # piped link [[Target|short(Target).lower()]] -> [[Target\x08]]
B_LP_LOW = b"\x0b"   # piped link [[Target|Target.lower()]] -> [[Target\x0b]]
B_AREA_M = b"\x0c"   # area km2 (mi2) with exact/delta mi2: <num><tag>\x0c<delta_char>
B_DENS_M = b"\x0e"   # density /km2 (/mi2) with exact/delta /mi2: <num>\x0e<delta_char>

RE_PAGE = re.compile(
    rb'    <title>(.*?)</title>\n'
    rb'    <id>(\d+)</id>\n'
    rb'(?:    <restrictions>(.*?)</restrictions>\n)?'
    rb'    <revision>\n'
    rb'      <id>(\d+)</id>\n'
    rb'      <timestamp>(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)</timestamp>\n'
    rb'      <contributor>\n'
    rb'(?:        <username>(.*?)</username>\n        <id>(\d+)</id>\n|        <ip>(.*?)</ip>\n)'
    rb'      </contributor>\n'
    rb'(      <minor />\n)?'
    rb'(?:      <comment>(.*?)</comment>\n)?'
    rb'      <text xml:space="preserve"(?: />|>(.*?)</text>)\n'
    rb'    </revision>\n'
    rb'  </page>\n',
    re.S,
)

RE_STD_REDIR = re.compile(rb'^#REDIRECT \[\[([^\n]+)\]\]$')
RE_PIPED_LINK = re.compile(rb'\[\[([^\]|\n\x01-\x08\x0b]+)\|([^\]|\n\x01-\x08\x0b]+)\]\]')
RE_PIPED_DEC = re.compile(rb'\[\[([^\]|\n]+)([\x07\x08\x0b])\]\]')

# Unit conversions in Rambot US Census pages (after &amp; -> & unescaping):
# Match plain "X.X km&sup2; (Y.Y mi&sup2;)" and linked "X.X [[square kilometre|km&sup2;]] (Y.Y [[square mile|mi&sup2;]])"
KM2_TO_MI2 = 1.0 / 2.589988110336
MI2_TO_KM2 = 2.589988110336
RE_AREA_ENC = re.compile(rb'([0-9,]+\.\d) (km&sup2;|\[\[square kilometre\|km&sup2;\]\]) \(([0-9,]+\.\d) (mi&sup2;|\[\[square mile\|mi&sup2;\]\])\)')
RE_AREA_DEC = re.compile(rb'([0-9,]+\.\d)\x0c([0-9a-z])')
RE_DENS_ENC = re.compile(rb'([0-9,]+\.\d)/km&sup2; \(([0-9,]+\.\d)/mi&sup2;\)')
RE_DENS_DEC = re.compile(rb'([0-9,]+\.\d)\x0e([0-9a-z])')


def fmt_num1(val, has_comma):
    s = f"{val:,.1f}" if has_comma else f"{val:.1f}"
    return s.encode("ascii")


def parse_ts(b):
    y, mo, d = int(b[0:4]), int(b[5:7]), int(b[8:10])
    h, mi, s = int(b[11:13]), int(b[14:16]), int(b[17:19])
    return calendar.timegm((y, mo, d, h, mi, s, 0, 0, 0))


def fmt_ts(sec):
    t = time.gmtime(sec)
    return f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}T{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}Z".encode("ascii")


def unescape_xml(b):
    return b.replace(b"&lt;", b"<").replace(b"&gt;", b">").replace(b"&quot;", b'"').replace(b"&amp;", b"&")


def escape_xml(b):
    return b.replace(b"&", b"&amp;").replace(b"<", b"&lt;").replace(b">", b"&gt;").replace(b'"', b"&quot;")


def short_title(title, min_len=4):
    idx = -1
    for sep in (b", ", b" ("):
        p = title.find(sep)
        if p >= min_len and (idx < 0 or p < idx):
            idx = p
    return title[:idx] if idx >= min_len else b""


def encode_links(b):
    def repl(m):
        tgt, disp = m.group(1), m.group(2)
        if disp == tgt.lower() and disp != tgt:
            return b"[[" + tgt + B_LP_LOW + b"]]"
        st = short_title(tgt, min_len=2)
        if st:
            if disp == st:
                return b"[[" + tgt + B_LP_SH + b"]]"
            if disp == st.lower() and disp != st:
                return b"[[" + tgt + B_LP_SHL + b"]]"
        return m.group(0)
    return RE_PIPED_LINK.sub(repl, b)


def decode_links(b):
    if B_LP_SH not in b and B_LP_SHL not in b and B_LP_LOW not in b:
        return b
    def repl(m):
        tgt, code = m.group(1), m.group(2)
        if code == B_LP_LOW:
            disp = tgt.lower()
        elif code == B_LP_SH:
            disp = short_title(tgt, min_len=2)
        else:
            disp = short_title(tgt, min_len=2).lower()
        return b"[[" + tgt + b"|" + disp + b"]]"
    return RE_PIPED_DEC.sub(repl, b)


def encode_units(b):
    if b"km&sup2;" not in b:
        return b
    def repl_area(m):
        a_s, ktag, b_s, mtag = m.groups()
        # Only handle matching plain/plain or linked/linked pairs
        is_linked = ktag.startswith(b"[[")
        if is_linked != mtag.startswith(b"[["):
            return m.group(0)
        a_val = float(a_s.replace(b",", b""))
        b_val = float(b_s.replace(b",", b""))
        pred10 = int(round(a_val * KM2_TO_MI2 * 10.0))
        b10 = int(round(b_val * 10.0))
        diff = b10 - pred10
        if -4 <= diff <= 4:
            # Verify exact string reconstruction including comma formatting
            has_c = (b"," in b_s)
            rec_b = fmt_num1((pred10 + diff) / 10.0, has_c)
            if rec_b == b_s and (has_c == (b10 >= 10000)):
                # Pack (is_linked, diff in -4..4) into a single ASCII char: '0'..'8' (plain) or 'a'..'i' (linked)
                ch = chr((ord("a") if is_linked else ord("0")) + (diff + 4)).encode("ascii")
                return a_s + B_AREA_M + ch
        return m.group(0)

    def repl_dens(m):
        a_s, b_s = m.groups()
        a_val = float(a_s.replace(b",", b""))
        b_val = float(b_s.replace(b",", b""))
        pred10 = int(round(a_val * MI2_TO_KM2 * 10.0))
        b10 = int(round(b_val * 10.0))
        diff = b10 - pred10
        if -9 <= diff <= 9:
            has_c = (b"," in b_s)
            rec_b = fmt_num1((pred10 + diff) / 10.0, has_c)
            if rec_b == b_s and (has_c == (b10 >= 10000)):
                # Encode diff in -9..9 into 'a'..'s'
                ch = chr(ord("a") + (diff + 9)).encode("ascii")
                return a_s + B_DENS_M + ch
        return m.group(0)

    b = RE_AREA_ENC.sub(repl_area, b)
    b = RE_DENS_ENC.sub(repl_dens, b)
    return b


def decode_units(b):
    if B_AREA_M in b:
        def repl_area(m):
            a_s, ch = m.groups()
            c = ch[0]
            if c >= ord("a"):
                is_linked = True
                diff = (c - ord("a")) - 4
            else:
                is_linked = False
                diff = (c - ord("0")) - 4
            a_val = float(a_s.replace(b",", b""))
            b10 = int(round(a_val * KM2_TO_MI2 * 10.0)) + diff
            b_s = fmt_num1(b10 / 10.0, b10 >= 10000)
            if is_linked:
                return a_s + b" [[square kilometre|km&sup2;]] (" + b_s + b" [[square mile|mi&sup2;]])"
            return a_s + b" km&sup2; (" + b_s + b" mi&sup2;)"
        b = RE_AREA_DEC.sub(repl_area, b)
    if B_DENS_M in b:
        def repl_dens(m):
            a_s, ch = m.groups()
            diff = (ch[0] - ord("a")) - 9
            a_val = float(a_s.replace(b",", b""))
            b10 = int(round(a_val * MI2_TO_KM2 * 10.0)) + diff
            b_s = fmt_num1(b10 / 10.0, b10 >= 10000)
            return a_s + b"/km&sup2; (" + b_s + b"/mi&sup2;)"
        b = RE_DENS_DEC.sub(repl_dens, b)
    return b


def encode_varints(arr):
    out = bytearray()
    for v in arr:
        v = int(v)
        assert v >= 0
        while v >= 0x80:
            out.append((v & 0x7F) | 0x80)
            v >>= 7
        out.append(v)
    return bytes(out)


def decode_varints(buf, pos, n):
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        v = 0
        sh = 0
        while True:
            b = buf[pos]
            pos += 1
            v |= (b & 0x7F) << sh
            if not (b & 0x80):
                break
            sh += 7
        out[i] = v
    return out, pos


def zigzag_enc(arr):
    a = np.asarray(arr, dtype=np.int64)
    return np.where(a >= 0, a * 2, -a * 2 - 1)


def zigzag_dec(arr):
    a = np.asarray(arr, dtype=np.int64)
    return np.where((a & 1) == 0, a >> 1, -((a + 1) >> 1))


def encode_enwik9(in_path="enwik9", out_prefix="enwik9.prep"):
    t0 = time.time()
    raw = open(in_path, "rb").read()
    orig_sha = hashlib.sha256(raw).hexdigest()
    print(f"Loaded {in_path} ({len(raw):,} B) in {time.time()-t0:.1f}s | sha256={orig_sha[:16]}...", flush=True)

    for cb in (b"\x00", B_ART, B_SEP, B_TITLE, B_STITLE, B_RSTD, B_RRAW, B_LP_SH, B_LP_SHL, B_LP_LOW, B_AREA_M, B_DENS_M):
        assert cb not in raw, f"Control byte {cb!r} found in {in_path}!"

    parts = raw.split(b"  <page>\n")
    header = parts[0]
    tail = b"  <page>\n" + parts[-1]
    page_chunks = parts[1:-1]
    n_pages = len(page_chunks)
    print(f"Parsing {n_pages:,} complete pages (header={len(header)}B, tail={len(tail)}B)...", flush=True)

    t1 = time.time()
    pids = np.empty(n_pages, dtype=np.int64)
    rids = np.empty(n_pages, dtype=np.int64)
    tss = np.empty(n_pages, dtype=np.int64)
    flags = np.empty(n_pages, dtype=np.uint8)
    contrib_raw = []
    comments = []
    restrs = []
    redir_stream = bytearray()
    census_stream = bytearray()
    art_stream = bytearray()

    n_redir = n_census = n_art = 0
    n_title_sub = n_stitle_sub = n_unesc_fallback = 0

    for i, pg in enumerate(page_chunks):
        m = RE_PAGE.fullmatch(pg)
        assert m is not None, f"Failed to parse page {i}: {pg[:200]!r}"
        title, pid_b, restr, rid_b, ts_b, uname, uid_b, ip, minor, comment, body = m.groups()
        pids[i] = int(pid_b)
        rids[i] = int(rid_b)
        tss[i] = parse_ts(ts_b)
        ckey = (b"U", uname, uid_b) if uname is not None else (b"I", ip, b"")
        contrib_raw.append(ckey)

        is_empty = body is None
        is_redir = False
        is_census = False
        body_unesc_ok = False
        if not is_empty:
            u = unescape_xml(body)
            if escape_xml(u) == body:
                body_u = u
                body_unesc_ok = True
            else:
                body_u = body
                n_unesc_fallback += 1
            if body_u[:9].lower() == b"#redirect":
                is_redir = True
            elif b"United States Census Bureau" in body_u or b"The racial makeup of the" in body_u:
                is_census = True

        fl = (
            (1 if minor else 0)
            | (2 if comment is not None else 0)
            | (4 if restr is not None else 0)
            | (8 if is_empty else 0)
            | (16 if is_redir else 0)
            | (32 if body_unesc_ok else 0)
            | (64 if is_census else 0)
        )
        flags[i] = fl
        if comment is not None:
            comments.append(comment)
        if restr is not None:
            restrs.append(restr)

        if is_empty:
            art_stream.extend(B_ART)
            art_stream.extend(title)
            art_stream.extend(B_SEP)
            n_art += 1
        elif is_redir:
            n_redir += 1
            mr = RE_STD_REDIR.fullmatch(body_u)
            if mr is not None:
                redir_stream.extend(B_RSTD)
                redir_stream.extend(title)
                redir_stream.extend(B_SEP)
                redir_stream.extend(mr.group(1))
            else:
                redir_stream.extend(B_RRAW)
                redir_stream.extend(title)
                redir_stream.extend(B_SEP)
                redir_stream.extend(body_u)
        else:
            # Order: links -> units -> exact title -> short title (decoder reverses exact order!)
            b_enc = encode_links(body_u)
            b_enc = encode_units(b_enc)
            if len(title) >= 4 and title in b_enc:
                n_title_sub += b_enc.count(title)
                b_enc = b_enc.replace(title, B_TITLE)
            st = short_title(title)
            if st and st in b_enc:
                n_stitle_sub += b_enc.count(st)
                b_enc = b_enc.replace(st, B_STITLE)

            dst = census_stream if is_census else art_stream
            if is_census:
                n_census += 1
            else:
                n_art += 1
            dst.extend(B_ART)
            dst.extend(title)
            dst.extend(B_SEP)
            dst.extend(b_enc)

    print(
        f"Parsed in {time.time()-t1:.1f}s | articles={n_art:,}, census={n_census:,}, redirects={n_redir:,} | "
        f"title_subs={n_title_sub:,}, short_title_subs={n_stitle_sub:,}, unesc_fallback={n_unesc_fallback:,}",
        flush=True,
    )

    c_counts = Counter(contrib_raw)
    contrib_list = [k for k, _ in c_counts.most_common()]
    contrib_map = {k: idx for idx, k in enumerate(contrib_list)}
    cids = np.array([contrib_map[k] for k in contrib_raw], dtype=np.int64)

    dpids = np.diff(pids, prepend=0)
    order = np.argsort(rids, kind="mergesort")
    dtss = np.diff(tss[order], prepend=0)

    t2 = time.time()
    parts_meta = [
        ("dpids", encode_varints(zigzag_enc(dpids))),
        ("rids", rids.astype(np.uint32).tobytes()),
        ("dtss_by_rid", encode_varints(zigzag_enc(dtss))),
        ("cids", encode_varints(cids)),
        ("contrib_tbl", b"\x00".join(k[0] + b"\x01" + k[1] + b"\x01" + k[2] for k in contrib_list)),
        ("flags", flags.tobytes()),
        ("restrs", b"\x00".join(restrs)),
        ("comments", b"\x00".join(comments)),
        ("header", header),
        ("tail", tail),
    ]
    meta_bin = bytearray()
    meta_bin.extend(struct.pack("<II", n_pages, len(parts_meta)))
    col_stats = {}
    for name, blob in parts_meta:
        meta_bin.extend(struct.pack("<I", len(blob)))
        meta_bin.extend(blob)
        c_each = len(lzma.compress(blob, preset=6))
        col_stats[name] = dict(raw=len(blob), lzma6=c_each)
        print(f"  meta[{name:12s}]: raw={len(blob)/1e6:6.2f} MB -> lzma6={c_each/1e6:6.3f} MB")

    meta_lzma = lzma.compress(bytes(meta_bin), preset=9)
    with open(out_prefix + ".meta.lzma", "wb") as f:
        f.write(meta_lzma)
    print(f"Total packed metadata: raw={len(meta_bin)/1e6:.2f} MB -> lzma9={len(meta_lzma)/1e6:.3f} MB ({time.time()-t2:.1f}s)", flush=True)

    # Concatenate: [redirects] + [census] + [articles] (exact split lengths implicit from flags!)
    text_blob = bytes(redir_stream) + bytes(census_stream) + bytes(art_stream)
    with open(out_prefix, "wb") as f:
        f.write(text_blob)
    print(
        f"Wrote {out_prefix}: {len(text_blob):,} B ({len(text_blob)/1e6:.2f} MB; "
        f"redirects={len(redir_stream)/1e6:.2f} MB, census={len(census_stream)/1e6:.2f} MB, articles={len(art_stream)/1e6:.2f} MB) | "
        f"shrink={(1 - len(text_blob)/len(raw))*100:.2f}%",
        flush=True,
    )

    t3 = time.time()
    rec_sha = verify_roundtrip(out_prefix, orig_sha)
    print(f"VERIFIED BIT-EXACT SHA-256 ROUND-TRIP ({rec_sha[:16]}...) in {time.time()-t3:.1f}s!", flush=True)

    meta_json = dict(
        source=in_path,
        n_bytes_orig=len(raw),
        n_bytes_prep=len(text_blob),
        n_pages=n_pages,
        n_articles=n_art,
        n_census=n_census,
        n_redirects=n_redir,
        extra_overhead_bytes=len(meta_lzma),
        meta_lzma_bytes=len(meta_lzma),
        sha256=orig_sha,
        columns=col_stats,
    )
    with open(out_prefix + ".meta.json", "w") as f:
        json.dump(meta_json, f, indent=2)
    print(f"Saved {out_prefix}.meta.json (extra_overhead_bytes={len(meta_lzma):,})", flush=True)


def verify_roundtrip(prefix, expected_sha):
    meta_bin = lzma.decompress(open(prefix + ".meta.lzma", "rb").read())
    n_pages, n_parts = struct.unpack_from("<II", meta_bin, 0)
    pos = 8
    blobs = []
    for _ in range(n_parts):
        (ln,) = struct.unpack_from("<I", meta_bin, pos)
        pos += 4
        blobs.append(meta_bin[pos : pos + ln])
        pos += ln
    assert pos == len(meta_bin)

    b_dpids, b_rids, b_dtss, b_cids, b_ctbl, b_flags, b_restrs, b_comms, header, tail = blobs
    dpids_zz, _ = decode_varints(b_dpids, 0, n_pages)
    pids = np.cumsum(zigzag_dec(dpids_zz))
    rids = np.frombuffer(b_rids, dtype=np.uint32).astype(np.int64)
    dtss_zz, _ = decode_varints(b_dtss, 0, n_pages)
    order = np.argsort(rids, kind="mergesort")
    tss = np.empty(n_pages, dtype=np.int64)
    tss[order] = np.cumsum(zigzag_dec(dtss_zz))
    cids, _ = decode_varints(b_cids, 0, n_pages)
    contrib_list = [tuple(row.split(b"\x01")) for row in b_ctbl.split(b"\x00")]
    flags = np.frombuffer(b_flags, dtype=np.uint8)
    restrs = b_restrs.split(b"\x00") if b_restrs else []
    comments = b_comms.split(b"\x00") if b_comms else []

    n_census_expected = int(((flags & 64) != 0).sum())

    text = open(prefix, "rb").read()
    first_art = text.find(B_ART)
    assert first_art >= 0
    r_part = text[:first_art]
    ca_part = text[first_art:]

    r_entries = []
    if r_part:
        p = 0
        L = len(r_part)
        while p < L:
            kind = r_part[p : p + 1]
            assert kind in (B_RSTD, B_RRAW)
            p += 1
            sep = r_part.find(B_SEP, p)
            title = r_part[p:sep]
            p = sep + 1
            n1 = r_part.find(B_RSTD, p)
            n2 = r_part.find(B_RRAW, p)
            nxt = L if (n1 < 0 and n2 < 0) else (n2 if n1 < 0 else (n1 if n2 < 0 else min(n1, n2)))
            r_entries.append((kind, title, r_part[p:nxt]))
            p = nxt

    all_ca = []
    for chunk in ca_part.split(B_ART)[1:]:
        sep = chunk.find(B_SEP)
        all_ca.append((chunk[:sep], chunk[sep + 1 :]))
    c_entries = all_ca[:n_census_expected]
    a_entries = all_ca[n_census_expected:]

    h = hashlib.sha256()
    h.update(header)
    ri = ci = ai = ci_comm = ci_rest = 0
    for i in range(n_pages):
        fl = int(flags[i])
        is_minor = bool(fl & 1)
        has_comm = bool(fl & 2)
        has_rest = bool(fl & 4)
        is_empty = bool(fl & 8)
        is_redir = bool(fl & 16)
        unesc_ok = bool(fl & 32)
        is_census = bool(fl & 64)

        if is_redir:
            kind, title, payload = r_entries[ri]
            ri += 1
            body_u = (b"#REDIRECT [[" + payload + b"]]") if kind == B_RSTD else payload
            body = escape_xml(body_u) if unesc_ok else body_u
        else:
            if is_census:
                title, b_enc = c_entries[ci]
                ci += 1
            else:
                title, b_enc = a_entries[ai]
                ai += 1
            if is_empty:
                body = None
            else:
                st = short_title(title)
                body_u = b_enc
                if st and B_STITLE in body_u:
                    body_u = body_u.replace(B_STITLE, st)
                if B_TITLE in body_u:
                    body_u = body_u.replace(B_TITLE, title)
                body_u = decode_units(body_u)
                body_u = decode_links(body_u)
                body = escape_xml(body_u) if unesc_ok else body_u

        ck = contrib_list[int(cids[i])]
        if ck[0] == b"U":
            c_xml = b"        <username>" + ck[1] + b"</username>\n        <id>" + ck[2] + b"</id>\n"
        else:
            c_xml = b"        <ip>" + ck[1] + b"</ip>\n"

        pg = bytearray(b"  <page>\n    <title>")
        pg.extend(title)
        pg.extend(b"</title>\n    <id>")
        pg.extend(str(int(pids[i])).encode("ascii"))
        pg.extend(b"</id>\n")
        if has_rest:
            pg.extend(b"    <restrictions>")
            pg.extend(restrs[ci_rest])
            ci_rest += 1
            pg.extend(b"</restrictions>\n")
        pg.extend(b"    <revision>\n      <id>")
        pg.extend(str(int(rids[i])).encode("ascii"))
        pg.extend(b"</id>\n      <timestamp>")
        pg.extend(fmt_ts(int(tss[i])))
        pg.extend(b"</timestamp>\n      <contributor>\n")
        pg.extend(c_xml)
        pg.extend(b"      </contributor>\n")
        if is_minor:
            pg.extend(b"      <minor />\n")
        if has_comm:
            pg.extend(b"      <comment>")
            pg.extend(comments[ci_comm])
            ci_comm += 1
            pg.extend(b"</comment>\n")
        if is_empty:
            pg.extend(b'      <text xml:space="preserve" />\n')
        else:
            pg.extend(b'      <text xml:space="preserve">')
            pg.extend(body)
            pg.extend(b"</text>\n")
        pg.extend(b"    </revision>\n  </page>\n")
        h.update(pg)

    assert ri == len(r_entries) and ci == len(c_entries) and ai == len(a_entries)
    assert ci_comm == len(comments) and ci_rest == len(restrs)
    h.update(tail)
    rec_sha = h.hexdigest()
    assert rec_sha == expected_sha, f"SHA-256 MISMATCH: {rec_sha} != {expected_sha}"
    return rec_sha


if __name__ == "__main__":
    encode_enwik9()
