// How each function's lines group into blocks: [line count, title, picture].
// A { body } / { then } entry keeps one loop / if as-is and groups what is inside it.
// Pictures (components/BlockViz.jsx): config, requests, queues, sched, budget,
// pages, hash, pipeline, batch, model, sample, outputs, inbox.

export const BLOCKS = {
  main: [
    [2, "Parse args → Config", "config"],
    [1, "Build the Engine"],
    { then: [[3, "Generate offline, print", "outputs"]] },
    [3, "Serve over HTTP", "inbox"],
  ],
  "Engine.__init__": [
    [4, "Tokenizer + stop ids"],
    [2, "Model runner: weights, KV memory, CUDA graphs", "config"],
    [4, "Page manager, scheduler, empty pipeline", "pages"],
  ],
  build_runner: [[4, "Load model, size the KV cache, capture graphs", "config"]],
  "ModelRunner.__init__": [
    [3, "Load weights, read shapes"],
    [4, "Fixed-address GPU buffers", "pipeline"],
  ],
  "ModelRunner.allocate_kv_cache": [
    [3, "Measure free GPU memory"],
    [2, "How many pages fit", "config"],
    [2, "One K/V pool per layer", "config"],
  ],
  "ModelRunner.capture_cuda_graphs": [
    [2, "List graph sizes", "config"],
    [1, "Record one CUDA graph per size", "config"],
  ],
  "KVCacheManager.__init__": [[5, "Every page free, no hashes yet", "pages"]],

  "Engine.generate": [
    [2, "Queue every prompt", "queues"],
    [2, "Step until no work is left", "pipeline"],
    [1, "Return outputs in order", "outputs"],
  ],
  "Engine.add_request": [
    [3, "Tokenize + check length", "requests"],
    [4, "Create Request → waiting queue", "queues"],
  ],
  "Engine.step": [
    [2, "1 · Launch the forward prepared last step", "pipeline"],
    [2, "2 · Finish the step before it", "outputs"],
    [1, "3 · Sample, append ? placeholders", "sample"],
    [2, "4 · Schedule the next batch", "sched"],
    [2, "5 · Prepare its inputs while the GPU runs", "batch"],
  ],
  "Engine._finish_step": [
    [3, "Wait for the sampled tokens on the CPU", "pipeline"],
    [2, "Per request: fill ?, stop check, detokenize", "outputs"],
  ],
  "Detokenizer.update": [[7, "Decode only the new text", "outputs"]],

  "Scheduler.schedule": [
    [4, "Start with the full token budget", "budget"],
    [1, "1 · Running first (preempt if out of pages)", "sched"],
    [1, "2 · Admit waiting (FIFO, prefix cache)", "sched"],
    [3, "3 · Build the batch", "budget"],
  ],
  "Scheduler.finish": [[3, "Mark finished, free its pages", "queues"]],

  "KVCacheManager._update_hashes": [[2, "Hash each full block (chained)", "hash"]],
  "KVCacheManager.find_cached_prefix": [
    [3, "Hash the prompt's full blocks", "hash"],
    [3, "Look up hashes until a miss", "hash"],
  ],
  "KVCacheManager.allocate": [
    [4, "How many new pages?", "pages"],
    [1, "Not enough free → False", "pages"],
    [1, "Reuse the cached pages", "pages"],
    [1, "Take fresh pages (evict old hash)", "pages"],
    [2, "Register full blocks", "hash"],
  ],
  "KVCacheManager._register_full_blocks": [[3, "Publish full blocks in hash_to_block", "hash"]],
  "KVCacheManager.free_request": [[2, "Drop refs, zero-ref pages → free list", "pages"]],

  "ModelRunner.prepare": [
    [3, "Decode graph fits?", "batch"],
    [3, "Mixed graph fits?", "batch"],
    [1, "Else: eager", "batch"],
  ],
  "ModelRunner._pack": [
    [3, "Empty lists + block table", "batch"],
    [1, "Per request: ids, positions, slots, block row", "batch"],
    [4, "Pad, return PackedBatch", "batch"],
  ],
  "ModelRunner.launch_forward": [[4, "Run the model, logits of each last row", "model"]],
  "ModelRunner.forward_step": [
    [1, "Graph: upload + replay", "batch"],
    [2, "Eager: upload + run", "batch"],
  ],
  "ModelRunner._upload": [
    [3, "Copy into GPU buffers, fill ? from prev_tokens", "batch"],
    [3, "Attention metadata", "batch"],
  ],

  "Model.forward": [
    [2, "Embed + RoPE rows", "model"],
    [1, "Each decoder layer", "model"],
    [1, "Final norm", "model"],
  ],
  "DecoderLayer.forward": [
    [2, "Norm + attention", "model"],
    [1, "MLP", "model"],
  ],
  "Attention.forward": [[3, "project → attend → output", "model"]],
  "Attention.project": [[4, "q, k, v + RoPE", "model"]],
  "Attention.attend": [
    [4, "Write new K/V into pages", "pages"],
    [1, "Attend over the block table", "model"],
  ],
  "Attention.output": [[1, "o_proj", "model"]],
  "MLP.forward": [[3, "gate · up → down", "model"]],

  "ModelRunner.launch_sample": [
    [2, "Pick next tokens", "sample"],
    [4, "Keep on GPU + async copy to CPU", "pipeline"],
  ],
  sample: [
    [1, "Repetition penalty", "sample"],
    [3, "Greedy argmax", "sample"],
  ],
  apply_repetition_penalty: [
    [3, "Rows that want a penalty", "sample"],
    [7, "Scale seen-token logits", "sample"],
  ],

  "AsyncEngine.__init__": [[5, "Inbox, streams, start engine thread", "inbox"]],
  "uvicorn.run": [[2, "Serve HTTP forever", "inbox"]],
  "AsyncEngine._loop": [
    { body: [
      [2, "Drain the inbox", "inbox"],
      [1, "One engine step", "pipeline"],
      [2, "Push outputs to clients", "inbox"],
    ] },
  ],
  "AsyncEngine.generate": [
    [5, "Send the request to the engine", "inbox"],
    [1, "Stream tokens back", "inbox"],
    [2, "Close the stream", "inbox"],
  ],
};
