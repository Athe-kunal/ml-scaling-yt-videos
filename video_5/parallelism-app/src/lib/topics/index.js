import { STEPS as NO_PARALLELISM_STEPS } from "./no_parallelism";
import { STEPS as DP_STEPS } from "./dp";
import { STEPS as ZERO1_STEPS } from "./zero1";
import { STEPS as ZERO2_STEPS } from "./zero2";
import { STEPS as ZERO3_STEPS } from "./zero3";
import { STEPS as FSDP_STEPS } from "./fsdp";
import { STEPS as TP_STEPS } from "./tp";
import { STEPS as FSDP_TP_STEPS } from "./fsdp_tp";

export const TOPICS = [
  { id: "NONE", label: "No Parallelism", sub: "one device · baseline · per pseudocode line", steps: NO_PARALLELISM_STEPS },
  { id: "DP", label: "Data Parallel", sub: "replicated model · AllReduce", steps: DP_STEPS },
  { id: "ZERO1", label: "ZeRO-1", sub: "+ shard optimizer state", steps: ZERO1_STEPS },
  { id: "ZERO2", label: "ZeRO-2", sub: "+ shard gradients", steps: ZERO2_STEPS },
  { id: "ZERO3", label: "ZeRO-3", sub: "+ shard weights", steps: ZERO3_STEPS },
  { id: "FSDP", label: "FSDP", sub: "book notation · per pseudocode line", steps: FSDP_STEPS },
  { id: "TP", label: "Tensor Parallel", sub: "AllGather + ReduceScatter · per pseudocode line", steps: TP_STEPS },
  { id: "FSDP_TP", label: "FSDP + TP", sub: "2D mesh (X, Y) · per pseudocode line", steps: FSDP_TP_STEPS },
  { id: "FLOPS_COMMS", label: "FLOPs vs Comms", sub: "mixed FSDP+TP · interactive chart", custom: "flops-comms" },
];

export const DEFAULT_TOPIC_ID = "NONE";

export function resolveTopicId(id) {
  return TOPICS.some((t) => t.id === id) ? id : DEFAULT_TOPIC_ID;
}

export function getTopic(id) {
  return TOPICS.find((t) => t.id === id) || TOPICS[0];
}
