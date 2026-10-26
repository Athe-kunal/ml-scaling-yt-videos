import { blockHash, call, emit, enter, leave, short } from "./trace.js";

// Line-for-line port of tinyserve.KVCacheManager. Tags in the emit() calls
// match the #@tags in codeSources.js (KV_SRC).

export class KV {
  constructor(w, numBlocks, bs, caching = true) {
    this.w = w;
    this.bs = bs;
    this.caching = caching;
    this.ref = Array(numBlocks).fill(0);
    this.bh = Array(numBlocks).fill(null);
    this.free = Array.from({ length: numBlocks - 1 }, (_, i) => i + 1); // OrderedDict order, front = LRU
    this.h2b = new Map();
    w.kv = this;
    w.pages = Array.from({ length: numBlocks }, () => Array(bs).fill(null));
  }

  get numFree() {
    return this.free.length;
  }

  updateHashes(req) {
    const w = this.w, bs = this.bs;
    const f = enter(w, "_update_hashes", { bs });
    while (req.block_hashes.length < Math.floor(req.token_ids.length / bs)) {
      const i = req.block_hashes.length;
      const parent = i ? req.block_hashes[i - 1] : null;
      const toks = req.token_ids.slice(i * bs, (i + 1) * bs);
      const h = blockHash(w, parent, toks);
      req.block_hashes.push(h);
      Object.assign(f.vars, { i, parent, h });
      emit(w, {
        code: "kv", tags: ["uh_loop", "uh_body"], kind: "hash", req: req.id, hashes: [h], hashBlock: { rid: req.id, i },
        notes: [{ tag: "uh_body", t: `${req.id}.h${i} = hash(${short(parent)}, ${toks.join(" ")}) = ${short(h)}` }],
      });
    }
    leave(w);
  }

  findCachedPrefix(req) {
    const w = this.w, bs = this.bs;
    const f = enter(w, "find_cached_prefix", { req: req.id });
    if (!this.caching) {
      leave(w);
      return [];
    }
    call(w, "kv", "fcp_hash", () => this.updateHashes(req));
    const maxBlocks = Math.floor((req.token_ids.length - 1) / bs);
    const hits = [];
    Object.assign(f.vars, { max_blocks: maxBlocks, hits });
    emit(w, {
      code: "kv", tags: ["fcp_max"], kind: "kv", req: req.id,
      notes: [{ tag: "fcp_max", t: `(${req.token_ids.length} - 1) // ${bs} = ${maxBlocks} blocks max` }],
    });
    for (const h of req.block_hashes.slice(0, maxBlocks)) {
      const b = this.h2b.get(h);
      Object.assign(f.vars, { h, b: b ?? null });
      if (b === undefined) {
        emit(w, {
          code: "kv", tags: ["fcp_loop", "fcp_miss"], kind: "miss", req: req.id, lookup: { h, hit: false },
          hashBlock: { rid: req.id, i: hits.length },
          notes: [{ tag: "fcp_loop", t: `${short(h)} not in hash_to_block → stop` }],
        });
        break;
      }
      hits.push(b);
      emit(w, {
        code: "kv", tags: ["fcp_loop", "fcp_hit"], kind: "hit", req: req.id, lookup: { h, hit: true }, pages: { [b]: "hit" },
        hashBlock: { rid: req.id, i: hits.length - 1 },
        notes: [{ tag: "fcp_loop", t: `${short(h)} → page ${b}` }],
      });
    }
    emit(w, { code: "kv", tags: ["fcp_ret"], kind: hits.length ? "hit" : "kv", req: req.id, notes: [{ tag: "fcp_ret", t: `hits = [${hits.join(", ")}]` }] });
    leave(w);
    return hits;
  }

  allocate(req, numNew, cached = []) {
    const w = this.w, bs = this.bs;
    const f = enter(w, "allocate", { req: req.id, num_new_tokens: numNew, cached: [...cached] });
    const total = req.num_computed + cached.length * bs + numNew;
    const tnb = Math.ceil(total / bs);
    const need = tnb - req.block_ids.length - cached.length;
    const reclaim = cached.filter((b) => this.ref[b] === 0).length;
    Object.assign(f.vars, { total, total_needed_blocks: tnb, need, reclaim, num_free: this.numFree });
    emit(w, {
      code: "kv", tags: ["al_need"], kind: "kv", req: req.id,
      notes: [
        { tag: "al_need", t: `${req.num_computed} + ${cached.length}·${bs} + ${numNew} = ${total}` },
        { tag: "al_tnb", t: `ceil(${total}/${bs}) = ${tnb}` },
        { tag: "al_needv", t: `${tnb} - ${req.block_ids.length} - ${cached.length} = ${need}` },
        { tag: "al_rec", t: `${reclaim}` },
      ],
    });
    if (need + reclaim > this.numFree) {
      emit(w, {
        code: "kv", tags: ["al_check", "al_fail"], kind: "fail", req: req.id,
        notes: [{ tag: "al_check", t: `${need} + ${reclaim} > ${this.numFree} free → False` }],
      });
      leave(w);
      return false;
    }
    emit(w, {
      code: "kv", tags: ["al_check"], kind: "kv", req: req.id,
      notes: [{ tag: "al_check", t: `${need} + ${reclaim} ≤ ${this.numFree} free ✓` }],
    });

    for (const b of cached) {
      const wasFree = this.ref[b] === 0;
      if (wasFree) this.free.splice(this.free.indexOf(b), 1);
      this.ref[b] += 1;
      req.block_ids.push(b);
      f.vars.b = b;
      emit(w, {
        code: "kv", tags: ["al_claim"], kind: "hit", req: req.id, pages: { [b]: "claim" },
        notes: [{ tag: "al_claim", t: wasFree ? `page ${b}: off free list, ref → ${this.ref[b]}` : `page ${b}: shared, ref → ${this.ref[b]}` }],
      });
    }

    for (let k = 0; k < Math.max(need, 0); k++) {
      const b = this.free.shift();
      f.vars.b = b;
      const old = this.bh[b];
      if (old !== null) {
        this.h2b.delete(old);
        this.bh[b] = null;
      }
      this.ref[b] = 1;
      req.block_ids.push(b);
      w.pages[b] = w.pages[b].map((s) => s && { ...s, stale: true });
      emit(w, {
        code: "kv", tags: old !== null ? ["al_new", "al_evict"] : ["al_new"], kind: old !== null ? "evict" : "kv", req: req.id,
        pages: { [b]: old !== null ? "evict" : "new" }, evicted: old !== null ? [old] : [],
        notes: [{ tag: "al_new", t: `popitem → page ${b}${old !== null ? ` · evict ${short(old)}` : ""}` }],
      });
    }

    call(w, "kv", "al_reg", () => this.registerFullBlocks(req, total));
    emit(w, { code: "kv", tags: ["al_ok"], kind: "kv", req: req.id, notes: [{ tag: "al_ok", t: `block_ids = [${req.block_ids.join(", ")}]` }] });
    leave(w);
    return true;
  }

  registerFullBlocks(req, numTokens) {
    const w = this.w, bs = this.bs;
    if (!this.caching) return;
    const f = enter(w, "_register_full_blocks", { num_tokens: numTokens });
    call(w, "kv", "reg_hash", () => this.updateHashes(req));
    const lim = Math.floor(Math.min(numTokens, req.token_ids.length) / bs);
    for (let i = 0; i < lim; i++) {
      const b = req.block_ids[i], h = req.block_hashes[i];
      if (this.bh[b] === null) {
        this.bh[b] = h;
        if (!this.h2b.has(h)) this.h2b.set(h, b);
        Object.assign(f.vars, { i, b, h });
        emit(w, {
          code: "kv", tags: ["reg_loop", "reg_set"], kind: "hash", req: req.id, pages: { [b]: "hash" }, hashes: [h],
          hashBlock: { rid: req.id, i },
          notes: [{ tag: "reg_set", t: `page ${b} ↔ ${short(h)}` }],
        });
      }
    }
    leave(w);
  }

  freeRequest(req) {
    const w = this.w;
    const f = enter(w, "free_request", { req: req.id });
    for (const b of [...req.block_ids].reverse()) {
      this.ref[b] -= 1;
      const freed = this.ref[b] === 0;
      if (freed) this.free.push(b);
      f.vars.b = b;
      emit(w, {
        code: "kv", tags: freed ? ["fr_loop", "fr_free"] : ["fr_loop"], kind: "free", req: req.id, pages: { [b]: freed ? "free" : "deref" },
        notes: [{ tag: "fr_loop", t: `page ${b}: ref → ${this.ref[b]}${freed ? " · to free tail (hash kept)" : " · still shared"}` }],
      });
    }
    req.block_ids = [];
    emit(w, { code: "kv", tags: ["fr_done"], kind: "free", req: req.id, notes: [{ tag: "fr_done", t: `${req.id}.block_ids = []` }] });
    leave(w);
  }
}
